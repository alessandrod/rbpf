use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use byteorder::{LittleEndian, WriteBytesExt};

use super::JitCodeMeta;

const ELF_CLASS_64: u8 = 2;
const ELF_DATA_2LSB: u8 = 1;
const ELF_VERSION: u32 = 1;
const ELF_OSABI_SYSV: u8 = 0;

const ET_EXEC: u16 = 2;
const EM_X86_64: u16 = 62;

const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;

const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;

const STB_GLOBAL: u8 = 1;
const STT_FUNC: u8 = 2;

const DW_LNS_COPY: u8 = 0x01;
const DW_LNS_ADVANCE_LINE: u8 = 0x03;
const DW_LNE_END_SEQUENCE: u8 = 0x01;
const DW_LNE_SET_ADDRESS: u8 = 0x02;

#[cfg_attr(test, allow(dead_code))]
const JIT_NOACTION: u32 = 0;
const JIT_REGISTER_FN: u32 = 1;
const JIT_UNREGISTER_FN: u32 = 2;

static JIT_GDB_LOCK: Mutex<()> = Mutex::new(());

pub struct JitGdbHook {
    entries: BTreeMap<u64, JitGdbEntry>,
}

impl JitGdbHook {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }
    pub fn on_code_load(
        &mut self,
        meta: &JitCodeMeta,
        text: &[u8],
        pc_offsets: &[u32],
        _program: &[u8],
    ) {
        let elf = build_elf(meta, text, pc_offsets);
        let mut entry = Box::new(JitCodeEntry {
            next: std::ptr::null_mut(),
            prev: std::ptr::null_mut(),
            symfile_addr: elf.as_ptr(),
            symfile_size: elf.len() as u64,
        });
        let entry_ptr = &mut *entry as *mut JitCodeEntry;
        unsafe {
            register_entry(entry_ptr);
        }
        self.entries
            .insert(meta.code_ptr, JitGdbEntry { entry, _elf: elf });
    }

    pub fn on_code_unload(&mut self, meta: &JitCodeMeta) {
        let entry = self.entries.remove(&meta.code_ptr);
        if let Some(mut entry) = entry {
            let entry_ptr = &mut *entry.entry as *mut JitCodeEntry;
            unsafe {
                unregister_entry(entry_ptr);
            }
        }
    }
}

struct JitGdbEntry {
    entry: Box<JitCodeEntry>,
    _elf: Vec<u8>,
}

#[repr(C)]
struct JitCodeEntry {
    next: *mut JitCodeEntry,
    prev: *mut JitCodeEntry,
    symfile_addr: *const u8,
    symfile_size: u64,
}

unsafe impl Send for JitCodeEntry {}
unsafe impl Sync for JitCodeEntry {}

#[repr(C)]
struct JitDescriptor {
    version: u32,
    action_flag: u32,
    relevant_entry: *mut JitCodeEntry,
    first_entry: *mut JitCodeEntry,
}

#[cfg(not(test))]
#[no_mangle]
static mut __jit_debug_descriptor: JitDescriptor = JitDescriptor {
    version: 1,
    action_flag: JIT_NOACTION,
    relevant_entry: std::ptr::null_mut(),
    first_entry: std::ptr::null_mut(),
};

#[cfg(test)]
extern "C" {
    static mut __jit_debug_descriptor: JitDescriptor;
}

#[cfg(not(test))]
#[no_mangle]
#[inline(never)]
pub extern "C" fn __jit_debug_register_code() {
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

#[cfg(test)]
extern "C" {
    fn __jit_debug_register_code();
}

unsafe fn register_entry(entry: *mut JitCodeEntry) {
    let _guard = JIT_GDB_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let descriptor = &raw mut __jit_debug_descriptor;
    (*entry).next = (*descriptor).first_entry;
    (*entry).prev = std::ptr::null_mut();
    if !(*descriptor).first_entry.is_null() {
        (*(*descriptor).first_entry).prev = entry;
    }
    (*descriptor).first_entry = entry;
    (*descriptor).relevant_entry = entry;
    (*descriptor).action_flag = JIT_REGISTER_FN;
    __jit_debug_register_code();
}

unsafe fn unregister_entry(entry: *mut JitCodeEntry) {
    let _guard = JIT_GDB_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let descriptor = &raw mut __jit_debug_descriptor;
    let prev = (*entry).prev;
    let next = (*entry).next;
    if !prev.is_null() {
        (*prev).next = next;
    } else {
        (*descriptor).first_entry = next;
    }
    if !next.is_null() {
        (*next).prev = prev;
    }
    (*descriptor).relevant_entry = entry;
    (*descriptor).action_flag = JIT_UNREGISTER_FN;
    __jit_debug_register_code();
}

struct Section {
    name: &'static str,
    sh_type: u32,
    sh_flags: u64,
    sh_addr: u64,
    sh_offset: u64,
    sh_addralign: u64,
    sh_entsize: u64,
    link: u32,
    info: u32,
    name_offset: u32,
    data: Vec<u8>,
}

fn build_elf(meta: &JitCodeMeta, text: &[u8], pc_offsets: &[u32]) -> Vec<u8> {
    let text_addr = meta.code_ptr;
    let mut strtab = Vec::new();
    strtab.push(0);

    let mut symtab = Vec::new();
    symtab.extend_from_slice(&[0u8; 24]);
    let text_section_index = 1u16;

    let mut emitted_symbols = BTreeSet::new();
    for symbol in &meta.symbols {
        if symbol.pc >= pc_offsets.len() {
            continue;
        }
        if !emitted_symbols.insert((symbol.pc, symbol.name.as_str())) {
            continue;
        }
        let Some(host_offset) = function_entry_offset(meta, text, pc_offsets, symbol.pc) else {
            continue;
        };
        let addr = text_addr.wrapping_add(host_offset as u64);
        let name_offset = add_str(&mut strtab, symbol.name.as_bytes());
        write_sym(
            &mut symtab,
            name_offset,
            STB_GLOBAL,
            STT_FUNC,
            text_section_index,
            addr,
            0,
        );
    }

    for symbol in &meta.host_symbols {
        if symbol.host_end <= symbol.host_start {
            continue;
        }
        let addr = text_addr.wrapping_add(symbol.host_start);
        let size = symbol.host_end.saturating_sub(symbol.host_start);
        let name_offset = add_str(&mut strtab, symbol.name.as_bytes());
        write_sym(
            &mut symtab,
            name_offset,
            STB_GLOBAL,
            STT_FUNC,
            text_section_index,
            addr,
            size,
        );
    }

    let debug_line = build_debug_line(meta, pc_offsets);

    let mut sections = vec![
        Section {
            name: ".text",
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_EXECINSTR,
            sh_addr: text_addr,
            sh_offset: 0,
            sh_addralign: 16,
            sh_entsize: 0,
            link: 0,
            info: 0,
            name_offset: 0,
            data: text.to_vec(),
        },
        Section {
            name: ".symtab",
            sh_type: SHT_SYMTAB,
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: 0,
            sh_addralign: 8,
            sh_entsize: 24,
            link: 3,
            info: 1,
            name_offset: 0,
            data: symtab,
        },
        Section {
            name: ".strtab",
            sh_type: SHT_STRTAB,
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: 0,
            sh_addralign: 1,
            sh_entsize: 0,
            link: 0,
            info: 0,
            name_offset: 0,
            data: strtab,
        },
        Section {
            name: ".debug_line",
            sh_type: SHT_PROGBITS,
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: 0,
            sh_addralign: 1,
            sh_entsize: 0,
            link: 0,
            info: 0,
            name_offset: 0,
            data: debug_line,
        },
        Section {
            name: ".shstrtab",
            sh_type: SHT_STRTAB,
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: 0,
            sh_addralign: 1,
            sh_entsize: 0,
            link: 0,
            info: 0,
            name_offset: 0,
            data: Vec::new(),
        },
    ];

    let mut shstrtab = Vec::new();
    shstrtab.push(0);
    for section in &mut sections {
        section.name_offset = shstrtab.len() as u32;
        shstrtab.extend_from_slice(section.name.as_bytes());
        shstrtab.push(0);
    }
    if let Some(section) = sections.last_mut() {
        section.data = shstrtab;
    }

    let shstrndx = sections.len() as u16;

    let mut offset = 64usize;
    for section in &mut sections {
        offset = align_up(offset, section.sh_addralign as usize);
        section.sh_addr = match section.name {
            ".text" => text_addr,
            _ => 0,
        };
        section.link = match section.name {
            ".symtab" => 3,
            _ => section.link,
        };
        section.info = match section.name {
            ".symtab" => 1,
            _ => section.info,
        };
        section.sh_addralign = match section.name {
            ".text" => 16,
            ".symtab" => 8,
            _ => section.sh_addralign,
        };
        section.sh_entsize = match section.name {
            ".symtab" => 24,
            _ => section.sh_entsize,
        };
        section.sh_addr = match section.name {
            ".text" => text_addr,
            _ => section.sh_addr,
        };
        section.sh_flags = match section.name {
            ".text" => SHF_ALLOC | SHF_EXECINSTR,
            _ => section.sh_flags,
        };
        section.sh_type = match section.name {
            ".text" => SHT_PROGBITS,
            ".symtab" => SHT_SYMTAB,
            ".strtab" | ".shstrtab" => SHT_STRTAB,
            _ => section.sh_type,
        };
        section.sh_offset = offset as u64;
        offset = offset.saturating_add(section.data.len());
    }

    let shoff = align_up(offset, 8);
    let shnum = (sections.len() + 1) as u16;

    let mut elf = Vec::with_capacity(shoff + shnum as usize * 64);
    write_elf_header(&mut elf, shoff as u64, shnum, shstrndx);

    for section in &sections {
        let target_offset = section.sh_offset as usize;
        if elf.len() < target_offset {
            elf.resize(target_offset, 0);
        }
        elf.extend_from_slice(&section.data);
    }

    if elf.len() < shoff {
        elf.resize(shoff, 0);
    }

    write_shdr(&mut elf, &SectionHeader::default());
    for section in &sections {
        write_shdr(
            &mut elf,
            &SectionHeader {
                name: section.name_offset,
                sh_type: section.sh_type,
                sh_flags: section.sh_flags,
                sh_addr: section.sh_addr,
                sh_offset: section.sh_offset,
                sh_size: section.data.len() as u64,
                sh_link: section.link,
                sh_info: section.info,
                sh_addralign: section.sh_addralign,
                sh_entsize: section.sh_entsize,
            },
        );
    }

    elf
}

fn function_entry_offset(
    meta: &JitCodeMeta,
    text: &[u8],
    pc_offsets: &[u32],
    pc: usize,
) -> Option<u32> {
    if pc < meta.function_entry_offsets.len() {
        let offset = meta.function_entry_offsets[pc];
        if offset != u32::MAX {
            return Some(offset & 0x7fff_ffff);
        }
    }
    if pc < pc_offsets.len() {
        let offset = (pc_offsets[pc] & 0x7fff_ffff) as usize;
        if let Some(prologue) = find_frame_prologue(text, offset) {
            return Some(prologue as u32);
        }
        return Some(offset as u32);
    }
    None
}

fn find_frame_prologue(text: &[u8], offset: usize) -> Option<usize> {
    if offset >= text.len() {
        return None;
    }
    let start = offset.saturating_sub(32);
    let prologue = [0x55u8, 0x48, 0x89, 0xe5];
    for idx in (start..offset).rev() {
        if idx + prologue.len() <= text.len() && text[idx..idx + 4] == prologue {
            return Some(idx);
        }
    }
    None
}

#[derive(Default)]
struct SectionHeader {
    name: u32,
    sh_type: u32,
    sh_flags: u64,
    sh_addr: u64,
    sh_offset: u64,
    sh_size: u64,
    sh_link: u32,
    sh_info: u32,
    sh_addralign: u64,
    sh_entsize: u64,
}

fn write_elf_header(buf: &mut Vec<u8>, shoff: u64, shnum: u16, shstrndx: u16) {
    buf.extend_from_slice(&[
        0x7f,
        b'E',
        b'L',
        b'F',
        ELF_CLASS_64,
        ELF_DATA_2LSB,
        1,
        ELF_OSABI_SYSV,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]);
    let _ = buf.write_u16::<LittleEndian>(ET_EXEC);
    let _ = buf.write_u16::<LittleEndian>(EM_X86_64);
    let _ = buf.write_u32::<LittleEndian>(ELF_VERSION);
    let _ = buf.write_u64::<LittleEndian>(0);
    let _ = buf.write_u64::<LittleEndian>(0);
    let _ = buf.write_u64::<LittleEndian>(shoff);
    let _ = buf.write_u32::<LittleEndian>(0);
    let _ = buf.write_u16::<LittleEndian>(64);
    let _ = buf.write_u16::<LittleEndian>(0);
    let _ = buf.write_u16::<LittleEndian>(0);
    let _ = buf.write_u16::<LittleEndian>(64);
    let _ = buf.write_u16::<LittleEndian>(shnum);
    let _ = buf.write_u16::<LittleEndian>(shstrndx);
}

fn write_shdr(buf: &mut Vec<u8>, shdr: &SectionHeader) {
    let _ = buf.write_u32::<LittleEndian>(shdr.name);
    let _ = buf.write_u32::<LittleEndian>(shdr.sh_type);
    let _ = buf.write_u64::<LittleEndian>(shdr.sh_flags);
    let _ = buf.write_u64::<LittleEndian>(shdr.sh_addr);
    let _ = buf.write_u64::<LittleEndian>(shdr.sh_offset);
    let _ = buf.write_u64::<LittleEndian>(shdr.sh_size);
    let _ = buf.write_u32::<LittleEndian>(shdr.sh_link);
    let _ = buf.write_u32::<LittleEndian>(shdr.sh_info);
    let _ = buf.write_u64::<LittleEndian>(shdr.sh_addralign);
    let _ = buf.write_u64::<LittleEndian>(shdr.sh_entsize);
}

fn write_sym(
    symtab: &mut Vec<u8>,
    name_offset: u32,
    binding: u8,
    ty: u8,
    shndx: u16,
    value: u64,
    size: u64,
) {
    let _ = symtab.write_u32::<LittleEndian>(name_offset);
    let _ = symtab.write_u8((binding << 4) | (ty & 0xf));
    let _ = symtab.write_u8(0);
    let _ = symtab.write_u16::<LittleEndian>(shndx);
    let _ = symtab.write_u64::<LittleEndian>(value);
    let _ = symtab.write_u64::<LittleEndian>(size);
}

fn add_str(strtab: &mut Vec<u8>, name: &[u8]) -> u32 {
    let offset = strtab.len();
    strtab.extend_from_slice(name);
    strtab.push(0);
    offset as u32
}

fn build_debug_line(meta: &JitCodeMeta, pc_offsets: &[u32]) -> Vec<u8> {
    let filename = meta.id.as_bytes();
    let mut header = Vec::new();
    header.push(1);
    header.push(1);
    header.push(1);
    header.push((-5i8) as u8);
    header.push(14);
    header.push(13);
    header.extend_from_slice(&[0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1]);
    header.push(0);
    header.extend_from_slice(filename);
    header.push(0);
    write_uleb128(&mut header, 0);
    write_uleb128(&mut header, 0);
    write_uleb128(&mut header, 0);
    header.push(0);

    let mut program = Vec::new();
    let mut current_line: i64 = 1;
    for (pc, offset) in pc_offsets.iter().enumerate() {
        let host_offset = offset & 0x7fff_ffff;
        let addr = meta.code_ptr + host_offset as u64;
        program.push(0);
        write_uleb128(&mut program, 1 + 8);
        program.push(DW_LNE_SET_ADDRESS);
        program.extend_from_slice(&addr.to_le_bytes());
        program.push(DW_LNS_ADVANCE_LINE);
        write_sleb128(&mut program, pc as i64 - current_line);
        current_line = pc as i64;
        program.push(DW_LNS_COPY);
    }
    program.push(0);
    write_uleb128(&mut program, 1);
    program.push(DW_LNE_END_SEQUENCE);

    let unit_length = (2 + 4 + header.len() + program.len()) as u32;
    let mut data = Vec::new();
    let _ = data.write_u32::<LittleEndian>(unit_length);
    let _ = data.write_u16::<LittleEndian>(4);
    let _ = data.write_u32::<LittleEndian>(header.len() as u32);
    data.extend_from_slice(&header);
    data.extend_from_slice(&program);
    data
}

fn write_uleb128(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
}

fn write_sleb128(buf: &mut Vec<u8>, mut value: i64) {
    loop {
        let byte = (value & 0x7f) as u8;
        let sign = byte & 0x40;
        value >>= 7;
        let done = (value == 0 && sign == 0) || (value == -1 && sign != 0);
        buf.push(if done { byte } else { byte | 0x80 });
        if done {
            break;
        }
    }
}

fn align_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    (value + align - 1) / align * align
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit_debug::{JitHostSymbol, JitSymbol};
    use crate::program::SBPFVersion;
    use object::{Object, ObjectSection, ObjectSymbol};
    use std::collections::HashSet;
    use std::sync::{Arc, Barrier, Mutex};
    use std::thread;

    unsafe fn reset_descriptor() {
        let _guard = JIT_GDB_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let descriptor = &raw mut __jit_debug_descriptor;
        (*descriptor).action_flag = JIT_NOACTION;
        (*descriptor).relevant_entry = std::ptr::null_mut();
        (*descriptor).first_entry = std::ptr::null_mut();
    }

    unsafe fn validate_list_integrity(
        expected_max: usize,
        valid_entries: &HashSet<usize>,
    ) -> usize {
        let _guard = JIT_GDB_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let descriptor = &raw const __jit_debug_descriptor;
        let mut count = 0usize;
        let mut prev = std::ptr::null_mut();
        let mut current = (*descriptor).first_entry;
        while !current.is_null() {
            let current_addr = current as usize;
            assert!(
                valid_entries.contains(&current_addr),
                "jit-gdb list contains an unknown entry"
            );
            count = count.saturating_add(1);
            assert!(
                count <= expected_max,
                "jit-gdb list exceeds expected size (cycle or corruption)"
            );
            let current_ref = &*current;
            assert_eq!(current_ref.prev, prev, "jit-gdb list prev link mismatch");
            prev = current;
            current = current_ref.next;
        }
        count
    }

    unsafe fn list_is_empty() -> bool {
        let _guard = JIT_GDB_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let descriptor = &raw const __jit_debug_descriptor;
        (*descriptor).first_entry.is_null()
    }

    #[test]
    fn test_build_elf_sections_and_symbols() {
        let mut text = vec![0x90u8; 16];
        text[..4].copy_from_slice(&[0x55, 0x48, 0x89, 0xe5]);
        text[8..12].copy_from_slice(&[0x55, 0x48, 0x89, 0xe5]);
        let pc_offsets = vec![0u32, 4, 8, 12];
        let meta = JitCodeMeta {
            id: "jit_test_program".to_string(),
            code_ptr: 0x1000,
            code_len: text.len(),
            sbpf_version: SBPFVersion::V0,
            symbols: vec![
                JitSymbol {
                    key: 0,
                    name: "func_a".to_string(),
                    pc: 0,
                },
                JitSymbol {
                    key: 1,
                    name: "jit_test_program".to_string(),
                    pc: 2,
                },
                JitSymbol {
                    key: 2,
                    name: "jit_test_program".to_string(),
                    pc: 2,
                },
            ],
            syscalls: Vec::new(),
            host_symbols: vec![JitHostSymbol {
                name: "ANCHOR_TRACE".to_string(),
                host_start: 12,
                host_end: 16,
            }],
            function_entry_offsets: vec![0, u32::MAX, 8, u32::MAX],
        };

        let elf = build_elf(&meta, &text, &pc_offsets);
        let file = object::File::parse(elf.as_slice()).expect("failed to parse ELF");

        let text_section = file.section_by_name(".text").expect("missing .text");
        let text_data = text_section.data().expect("failed to read .text");
        assert_eq!(text_data, text.as_slice());

        let debug_line = file
            .section_by_name(".debug_line")
            .expect("missing .debug_line");
        let debug_line_data = debug_line.data().expect("failed to read .debug_line");
        assert!(
            !debug_line_data.is_empty(),
            ".debug_line should not be empty"
        );

        let symbols: Vec<_> = file
            .symbols()
            .filter(|symbol| !symbol.is_undefined())
            .map(|symbol| (symbol.name().unwrap(), symbol.address()))
            .collect();
        assert_eq!(
            symbols,
            vec![
                ("func_a", 0x1000),
                ("jit_test_program", 0x1008),
                ("ANCHOR_TRACE", 0x100c),
            ]
        );
    }

    #[test]
    fn test_register_unregister_multithreaded() {
        const THREADS: usize = 8;
        const ENTRIES_PER_THREAD: usize = 64;
        let expected = THREADS * ENTRIES_PER_THREAD;

        unsafe {
            reset_descriptor();
        }

        let registered_barrier = Arc::new(Barrier::new(THREADS + 1));
        let validated_barrier = Arc::new(Barrier::new(THREADS + 1));
        let entry_ptrs = Arc::new(Mutex::new(Vec::with_capacity(expected)));
        let mut handles = Vec::with_capacity(THREADS);

        for _ in 0..THREADS {
            let registered_barrier = Arc::clone(&registered_barrier);
            let validated_barrier = Arc::clone(&validated_barrier);
            let entry_ptrs = Arc::clone(&entry_ptrs);
            handles.push(thread::spawn(move || {
                let mut entries = Vec::with_capacity(ENTRIES_PER_THREAD);
                let mut local_ptrs = Vec::with_capacity(ENTRIES_PER_THREAD);
                for _ in 0..ENTRIES_PER_THREAD {
                    let mut entry = Box::new(JitCodeEntry {
                        next: std::ptr::null_mut(),
                        prev: std::ptr::null_mut(),
                        symfile_addr: std::ptr::null(),
                        symfile_size: 0,
                    });
                    let entry_ptr = &mut *entry as *mut JitCodeEntry;
                    unsafe {
                        register_entry(entry_ptr);
                    }
                    entries.push(entry);
                    local_ptrs.push(entry_ptr as usize);
                }
                {
                    let mut slot = entry_ptrs.lock().unwrap_or_else(|err| err.into_inner());
                    slot.extend(local_ptrs);
                }

                registered_barrier.wait();
                validated_barrier.wait();

                for mut entry in entries {
                    let entry_ptr = &mut *entry as *mut JitCodeEntry;
                    unsafe {
                        unregister_entry(entry_ptr);
                    }
                }
            }));
        }

        registered_barrier.wait();
        let ptrs_snapshot = {
            let slot = entry_ptrs.lock().unwrap_or_else(|err| err.into_inner());
            slot.clone()
        };
        assert_eq!(ptrs_snapshot.len(), expected);
        let ptr_set: HashSet<usize> = ptrs_snapshot.into_iter().collect();
        let count = unsafe { validate_list_integrity(expected, &ptr_set) };
        assert_eq!(count, expected);
        validated_barrier.wait();

        for handle in handles {
            handle.join().unwrap();
        }

        assert!(unsafe { list_is_empty() });
    }
}
