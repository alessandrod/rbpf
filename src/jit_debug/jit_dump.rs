use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{File, OpenOptions},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use byteorder::{LittleEndian, WriteBytesExt};

use crate::{
    disassembler::disassemble_instruction,
    ebpf,
    program::{BuiltinProgram, FunctionRegistry},
    static_analysis::CfgNode,
    vm::{ContextObject, EncryptedHostAddressToEbpfVm},
};

use crate::jit_debug::JitCodeMeta;

const JITDUMP_MAGIC: u32 = 0x4a695444;
const JITDUMP_VERSION: u32 = 1;
const JITDUMP_ELF_MACHINE: u32 = 62;
const RECORD_HEADER_SIZE: usize = 16;
const CODE_LOAD_ID: u32 = 0;
const CODE_MOVE_ID: u32 = 1;
const CODE_DEBUG_INFO_ID: u32 = 2;
const CODE_CLOSE_ID: u32 = 3;

static NEXT_DUMP_ID: AtomicU64 = AtomicU64::new(0);

pub struct JitDumpHook {
    dir: PathBuf,
    writer: Option<JitDumpWriter>,
}

impl JitDumpHook {
    pub fn from_env() -> Option<Self> {
        let dir = match std::env::var_os("JITDUMP_DIR") {
            Some(dir) => dir,
            None => return None,
        };
        Some(JitDumpHook {
            dir: PathBuf::from(dir),
            writer: None,
        })
    }

    pub fn on_code_load(
        &mut self,
        meta: &JitCodeMeta,
        text: &[u8],
        pc_offsets: &[u32],
        program: &[u8],
    ) {
        let functions = build_function_ranges(meta, text, pc_offsets);
        if functions.is_empty() {
            return;
        }
        let function_registry = build_function_registry(meta);
        let loader = build_syscall_loader(meta);
        let cfg_nodes = BTreeMap::new();
        let sanitized_id = sanitize_component(&meta.id);
        let mut writer = match JitDumpWriter::new(&self.dir, &sanitized_id) {
            Ok(writer) => writer,
            Err(err) => {
                log::warn!("Failed to create JIT dump for {}: {err}", meta.id);
                return;
            }
        };
        for symbol in &meta.host_symbols {
            if symbol.host_end <= symbol.host_start {
                continue;
            }
            let start = symbol.host_start as usize;
            let end = symbol.host_end as usize;
            if end <= start || end > text.len() {
                continue;
            }
            let code_slice = &text[start..end];
            let code_addr = meta.code_ptr + symbol.host_start;
            let _ = writer.write_code_load(&symbol.name, code_addr, code_slice);
        }
        for (range_index, function) in functions.into_iter().enumerate() {
            let start = function.host_start as usize;
            let end = function.host_end as usize;
            if end <= start || end > text.len() {
                continue;
            }
            let code_slice = &text[start..end];
            let code_addr = meta.code_ptr + function.host_start;
            let source_path = writer.source_path(range_index);
            match writer.create_source_file(&source_path).and_then(|file| {
                write_source_file(
                    file,
                    program,
                    function.pc_start,
                    function.pc_end,
                    &function_registry,
                    &loader,
                    meta.sbpf_version,
                    &cfg_nodes,
                )
            }) {
                Ok(()) => {
                    let _ = writer.write_debug_info_range(
                        code_addr,
                        function.host_start,
                        &source_path.to_string_lossy(),
                        function.pc_start,
                        function.pc_end,
                        pc_offsets,
                    );
                }
                Err(err) => {
                    log::warn!(
                        "Failed to write JIT source file {}: {err}",
                        source_path.display()
                    );
                }
            }
            let _ = writer.write_code_load(&function.name, code_addr, code_slice);
        }
        self.writer = Some(writer);
    }

    pub fn on_code_unload(&mut self) {
        self.writer.take();
    }
}

struct JitDumpWriter {
    file: Option<BufWriter<File>>,
    code_index: u64,
    pid: u32,
    dir: PathBuf,
    file_stem: String,
    dump_path: PathBuf,
    source_paths: Vec<PathBuf>,
}

impl JitDumpWriter {
    fn new(dir: &Path, id: &str) -> io::Result<Self> {
        let dir = std::path::absolute(dir)?;
        std::fs::create_dir_all(&dir)?;
        let pid = unsafe { libc::getpid() as u32 };
        let header = build_header(pid)?;
        let (file, file_stem, dump_path) = loop {
            let generation = NEXT_DUMP_ID.fetch_add(1, Ordering::Relaxed);
            let file_stem = format!("jit-{pid}-{id}-{generation}");
            let dump_path = dir.join(format!("{file_stem}.dump"));
            match OpenOptions::new()
                .create_new(true)
                .read(true)
                .write(true)
                .open(&dump_path)
            {
                Ok(file) => break (file, file_stem, dump_path),
                Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(err) => return Err(err),
            }
        };
        // Own the path before writing so initialization failures also clean it up.
        let mut writer = Self {
            file: None,
            code_index: 0,
            pid,
            dir,
            file_stem,
            dump_path,
            source_paths: Vec::new(),
        };
        let mut file = BufWriter::new(file);
        file.write_all(&header)?;
        writer.file = Some(file);
        Ok(writer)
    }

    fn write_code_load(&mut self, name: &str, code_addr: u64, code_bytes: &[u8]) -> io::Result<()> {
        let code_index = self.code_index;
        self.code_index = code_index.saturating_add(1);

        let pid = self.pid;
        let tid = get_tid();
        let code_len = code_bytes.len() as u64;
        let payload_len = 40 + name.len() + 1 + code_bytes.len();
        Self::write_record(
            self.file.as_mut().unwrap(),
            CODE_LOAD_ID,
            payload_len,
            |writer| {
                writer.write_u32::<LittleEndian>(pid)?;
                writer.write_u32::<LittleEndian>(tid)?;
                writer.write_u64::<LittleEndian>(code_addr)?;
                writer.write_u64::<LittleEndian>(code_addr)?;
                writer.write_u64::<LittleEndian>(code_len)?;
                writer.write_u64::<LittleEndian>(code_index)?;
                writer.write_all(name.as_bytes())?;
                writer.write_u8(0)?;
                writer.write_all(code_bytes)
            },
        )
    }

    fn write_debug_info_range(
        &mut self,
        code_addr: u64,
        function_host_start: u64,
        filename: &str,
        pc_start: usize,
        pc_end: usize,
        pc_offsets: &[u32],
    ) -> io::Result<()> {
        let filename = filename.as_bytes();
        let mut entries = Vec::new();
        let mut last_addr = None;
        for pc in pc_start..pc_end {
            let offset = pc_offsets[pc];
            let host_offset = offset & 0x7fff_ffff;
            let relative = (host_offset as u64).saturating_sub(function_host_start);
            let addr = code_addr.wrapping_add(relative);
            if last_addr.map_or(false, |last| addr <= last) {
                continue;
            }
            entries.push((addr, (pc - pc_start + 1) as u32));
            last_addr = Some(addr);
        }

        let per_entry_len = 8 + 4 + 4 + filename.len() + 1;
        let payload_len = 16 + entries.len() * per_entry_len;
        Self::write_record(
            self.file.as_mut().unwrap(),
            CODE_DEBUG_INFO_ID,
            payload_len,
            |writer| {
                writer.write_u64::<LittleEndian>(code_addr)?;
                writer.write_u64::<LittleEndian>(entries.len() as u64)?;
                for (addr, line) in entries.iter() {
                    writer.write_u64::<LittleEndian>(*addr)?;
                    writer.write_u32::<LittleEndian>(*line)?;
                    writer.write_u32::<LittleEndian>(0)?;
                    writer.write_all(filename)?;
                    writer.write_u8(0)?;
                }
                Ok(())
            },
        )?;
        Ok(())
    }

    fn write_record<F>(
        file: &mut BufWriter<File>,
        record_id: u32,
        payload_len: usize,
        write_payload: F,
    ) -> io::Result<()>
    where
        F: FnOnce(&mut BufWriter<File>) -> io::Result<()>,
    {
        let timestamp = timestamp();
        let mut total_size = RECORD_HEADER_SIZE + payload_len;
        let aligned_size = align_up(total_size, 8);
        let padding = aligned_size - total_size;
        total_size = aligned_size;

        file.write_u32::<LittleEndian>(record_id)?;
        file.write_u32::<LittleEndian>(total_size as u32)?;
        file.write_u64::<LittleEndian>(timestamp)?;
        write_payload(file)?;
        if padding != 0 {
            let padding_bytes = [0u8; 8];
            file.write_all(&padding_bytes[..padding])?;
        }
        // Flush so external tools can observe the record promptly.
        file.flush()
    }
}

impl JitDumpWriter {
    fn source_path(&self, range_index: usize) -> PathBuf {
        self.dir
            .join(format!("{}-{range_index}.sbpf", self.file_stem))
    }

    fn create_source_file(&mut self, path: &Path) -> io::Result<File> {
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        // Track ownership before writing, including sources that fail partway through.
        self.source_paths.push(path.to_path_buf());
        Ok(file)
    }
}

impl Drop for JitDumpWriter {
    fn drop(&mut self) {
        if let Some(mut file) = self.file.take() {
            if let Err(err) = Self::write_record(&mut file, CODE_CLOSE_ID, 0, |_| Ok(())) {
                log::warn!(
                    "Failed to close JIT dump {}: {err}",
                    self.dump_path.display()
                );
            }
        }
        // Close our handle before unlinking; existing readers can still consume CLOSE.
        for path in std::iter::once(&self.dump_path).chain(&self.source_paths) {
            if let Err(err) = std::fs::remove_file(path) {
                if err.kind() != io::ErrorKind::NotFound {
                    log::warn!(
                        "Failed to remove JIT dump artifact {}: {err}",
                        path.display()
                    );
                }
            }
        }
    }
}

fn sanitize_component(value: &str) -> String {
    let mut sanitized = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            sanitized.push(ch);
        } else {
            sanitized.push('_');
        }
    }
    if sanitized.is_empty() {
        sanitized.push_str("unknown");
    }
    sanitized
}

fn build_header(pid: u32) -> io::Result<Vec<u8>> {
    let timestamp = timestamp();
    let mut header = Vec::with_capacity(40);
    header.write_u32::<LittleEndian>(JITDUMP_MAGIC)?;
    header.write_u32::<LittleEndian>(JITDUMP_VERSION)?;
    header.write_u32::<LittleEndian>(40)?;
    header.write_u32::<LittleEndian>(JITDUMP_ELF_MACHINE)?;
    header.write_u32::<LittleEndian>(0)?;
    header.write_u32::<LittleEndian>(pid)?;
    header.write_u64::<LittleEndian>(timestamp)?;
    header.write_u64::<LittleEndian>(0)?;
    Ok(header)
}

fn timestamp() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
    }
    (ts.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(ts.tv_nsec as u64)
}

fn get_tid() -> u32 {
    unsafe { libc::syscall(libc::SYS_gettid) as u32 }
}

fn align_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    (value + align - 1) / align * align
}

#[allow(dead_code)]
fn _keep_constants() {
    let _ = CODE_MOVE_ID;
    let _ = CODE_CLOSE_ID;
}

struct FunctionRange {
    name: String,
    pc_start: usize,
    pc_end: usize,
    host_start: u64,
    host_end: u64,
}

fn build_function_ranges(
    meta: &JitCodeMeta,
    text: &[u8],
    pc_offsets: &[u32],
) -> Vec<FunctionRange> {
    let mut functions: BTreeMap<usize, BTreeSet<&str>> = BTreeMap::new();
    for symbol in &meta.symbols {
        if symbol.name.is_empty() || symbol.pc >= pc_offsets.len() {
            continue;
        }
        functions
            .entry(symbol.pc)
            .or_default()
            .insert(symbol.name.as_str());
    }
    let mut starts: Vec<usize> = functions.keys().copied().collect();
    starts.sort_unstable();
    if starts.is_empty() {
        return Vec::new();
    }

    let text_len = text.len();
    let mut ranges = Vec::new();
    for (idx, pc_start) in starts.iter().enumerate() {
        let pc_end = if idx + 1 < starts.len() {
            starts[idx + 1]
        } else {
            pc_offsets.len()
        };
        let host_start =
            function_entry_offset(meta, text, pc_offsets, *pc_start).unwrap_or(text_len as u64);
        let host_end = if pc_end < pc_offsets.len() {
            function_entry_offset(meta, text, pc_offsets, pc_end).unwrap_or(text_len as u64)
        } else {
            text_len as u64
        };
        if host_end <= host_start || host_end > text_len as u64 {
            continue;
        }
        if let Some(names) = functions.get(pc_start) {
            for name in names {
                ranges.push(FunctionRange {
                    name: name.to_string(),
                    pc_start: *pc_start,
                    pc_end,
                    host_start,
                    host_end,
                });
            }
        }
    }
    ranges
}

fn function_entry_offset(
    meta: &JitCodeMeta,
    text: &[u8],
    pc_offsets: &[u32],
    pc: usize,
) -> Option<u64> {
    if pc < meta.function_entry_offsets.len() {
        let offset = meta.function_entry_offsets[pc];
        if offset != u32::MAX {
            return Some((offset & 0x7fff_ffff) as u64);
        }
    }
    if pc < pc_offsets.len() {
        let offset = (pc_offsets[pc] & 0x7fff_ffff) as usize;
        if let Some(prologue) = find_frame_prologue(text, offset) {
            return Some(prologue as u64);
        }
        return Some(offset as u64);
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

fn write_source_file(
    file: File,
    program: &[u8],
    pc_start: usize,
    pc_end: usize,
    function_registry: &FunctionRegistry<usize>,
    loader: &BuiltinProgram<JitDumpContext>,
    sbpf_version: crate::program::SBPFVersion,
    cfg_nodes: &BTreeMap<usize, CfgNode>,
) -> io::Result<()> {
    if pc_end <= pc_start {
        return Ok(());
    }
    let mut writer = BufWriter::new(file);
    let mut lddw_continuation = false;
    for pc in pc_start..pc_end {
        let start = pc.saturating_mul(ebpf::INSN_SIZE);
        let end = start.saturating_add(ebpf::INSN_SIZE);
        if end > program.len() {
            break;
        }
        let mut bytes = [0u8; ebpf::INSN_SIZE];
        bytes.copy_from_slice(&program[start..end]);
        let disasm = if lddw_continuation {
            lddw_continuation = false;
            "lddw continuation".to_string()
        } else {
            let mut insn = ebpf::get_insn_unchecked(program, pc);
            if insn.opc == ebpf::LD_DW_IMM && !sbpf_version.disable_lddw() {
                if pc_end.saturating_sub(pc) < 2
                    || program.len().saturating_sub(end) < ebpf::INSN_SIZE
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "truncated lddw in JIT source",
                    ));
                }
                ebpf::augment_lddw_unchecked(program, &mut insn);
                lddw_continuation = true;
            }
            disassemble_instruction(
                &insn,
                pc,
                cfg_nodes,
                function_registry,
                loader,
                sbpf_version,
            )
        };
        writeln!(
            writer,
            "pc {:04}: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} | {}",
            pc,
            bytes[0],
            bytes[1],
            bytes[2],
            bytes[3],
            bytes[4],
            bytes[5],
            bytes[6],
            bytes[7],
            disasm
        )?;
    }
    writer.flush()
}

struct JitDumpContext;

impl ContextObject for JitDumpContext {
    fn consume(&mut self, _amount: u64) {}

    fn get_remaining(&self) -> u64 {
        0
    }

    fn active_mapping_ptr(&mut self) -> std::ptr::NonNull<crate::memory_region::MemoryMapping> {
        unreachable!("JitDumpContext is only used for disassembly")
    }
}

fn dummy_syscall(
    _vm: EncryptedHostAddressToEbpfVm<JitDumpContext>,
    _a: u64,
    _b: u64,
    _c: u64,
    _d: u64,
    _e: u64,
) {
}

fn build_syscall_loader(meta: &JitCodeMeta) -> BuiltinProgram<JitDumpContext> {
    let mut loader = BuiltinProgram::<JitDumpContext>::new_builtin();
    for syscall in &meta.syscalls {
        let _ = loader.register_function(&syscall.name, (dummy_syscall, |_| {}));
    }
    loader
}

fn build_function_registry(meta: &JitCodeMeta) -> FunctionRegistry<usize> {
    let mut registry = FunctionRegistry::default();
    for symbol in &meta.symbols {
        let _ = registry.register_function(symbol.key, symbol.name.as_bytes(), symbol.pc);
    }
    registry
}

#[cfg(test)]
mod tests {
    use crate::jit_debug::jit_dump::*;
    use crate::{
        elf::Executable, jit_debug::JitSymbol, program::SBPFVersion,
        static_analysis::DummyContextObject, verifier::RequisiteVerifier,
    };
    use byteorder::{ByteOrder, ReadBytesExt};
    use std::{
        io::Read,
        process::Command,
        sync::{Arc, Barrier},
        thread,
    };

    #[test]
    fn test_source_file_preserves_lddw_and_pc_lines() {
        let path = std::env::temp_dir().join(format!("sbpf-source-{}", rand::random::<u64>()));
        let program = [
            0x18, 0, 0, 0, 0x88, 0x77, 0x66, 0x55, 0, 0, 0, 0, 0x44, 0x33, 0x22, 0x11, 0x95, 0, 0,
            0, 0, 0, 0, 0,
        ];
        write_source_file(
            File::create(&path).unwrap(),
            &program,
            0,
            3,
            &FunctionRegistry::default(),
            &BuiltinProgram::new_builtin(),
            SBPFVersion::V0,
            &BTreeMap::new(),
        )
        .unwrap();
        let source = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            source,
            concat!(
                "pc 0000: 18 00 00 00 88 77 66 55 | lddw r0, 0x1122334455667788\n",
                "pc 0001: 00 00 00 00 44 33 22 11 | lddw continuation\n",
                "pc 0002: 95 00 00 00 00 00 00 00 | exit\n",
            )
        );
        for (program, pc_end) in [(&program[..8], 1), (&program[..], 1)] {
            let error = write_source_file(
                File::create(&path).unwrap(),
                program,
                0,
                pc_end,
                &FunctionRegistry::default(),
                &BuiltinProgram::new_builtin(),
                SBPFVersion::V0,
                &BTreeMap::new(),
            )
            .unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        }
        std::fs::remove_file(path).unwrap();
    }

    fn read_records(mut reader: impl Read) -> Vec<(u32, Vec<u8>)> {
        let mut data = Vec::new();
        reader.read_to_end(&mut data).unwrap();
        let mut data = data.as_slice();
        assert_eq!(data.read_u32::<LittleEndian>().unwrap(), JITDUMP_MAGIC);
        assert_eq!(data.read_u32::<LittleEndian>().unwrap(), JITDUMP_VERSION);
        assert_eq!(data.read_u32::<LittleEndian>().unwrap(), 40);
        let mut last_timestamp = LittleEndian::read_u64(&data[12..]);
        data = &data[28..];
        let mut records = Vec::new();
        while !data.is_empty() {
            let id = data.read_u32::<LittleEndian>().unwrap();
            let size = data.read_u32::<LittleEndian>().unwrap() as usize;
            let timestamp = data.read_u64::<LittleEndian>().unwrap();
            assert!(timestamp >= last_timestamp);
            last_timestamp = timestamp;
            assert!(size >= RECORD_HEADER_SIZE && size % 8 == 0);
            let mut payload = vec![0; size - RECORD_HEADER_SIZE];
            data.read_exact(&mut payload).unwrap();
            records.push((id, payload));
        }
        records
    }

    #[test]
    fn test_dump_files_survive_recompilation() {
        const CHILD_ENV: &str = "SBPF_JITDUMP_TEST_CHILD";
        if std::env::var_os(CHILD_ENV).is_none() {
            let dir = std::env::temp_dir().join(format!("sbpf-jitdump-{}", rand::random::<u64>()));
            std::fs::create_dir(&dir).unwrap();
            // Isolate both JITDUMP_DIR and the generation counter from other tests.
            let output = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "jit_debug::jit_dump::tests::test_dump_files_survive_recompilation",
                ])
                .env(CHILD_ENV, "1")
                .env("JITDUMP_DIR", &dir)
                .output()
                .unwrap();
            std::fs::remove_dir_all(dir).unwrap();
            assert!(
                output.status.success(),
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        const PROGRAM_ID: &str = "11157t3sqMV725NVRLrVQbAu98Jjfk1uCKehJnXXQs";
        const OTHER_PROGRAM_ID: &str = "1117mWrzzrZr312ebPDHu8tbfMwFNvCvMbr6WepCNG";
        let dir = PathBuf::from(std::env::var_os("JITDUMP_DIR").unwrap());
        let pid = std::process::id();
        let existing_dump = dir.join(format!("jit-{pid}-{PROGRAM_ID}-0.dump"));
        let existing_source = dir.join(format!("jit-{pid}-{PROGRAM_ID}-1-0.sbpf"));
        std::fs::write(&existing_dump, b"existing dump").unwrap();
        std::fs::write(&existing_source, b"existing source").unwrap();

        let compile = |program_id: &str, value: u8| {
            let mut program = Vec::new();
            let mut registry = FunctionRegistry::default();
            // Exercise shared names across programs and names that sanitize identically.
            for (index, name) in ["entrypoint", "shared", "shared/name", "shared?name"]
                .iter()
                .enumerate()
            {
                registry
                    .register_function(index as u32, name.as_bytes(), index * 2)
                    .unwrap();
                program.extend_from_slice(&[ebpf::MOV64_IMM, 0, 0, 0, value, 0, 0, 0]);
                program.extend_from_slice(&[ebpf::EXIT, 0, 0, 0, 0, 0, 0, 0]);
            }
            let mut executable = Executable::<DummyContextObject>::from_text_bytes(
                &program,
                Arc::new(BuiltinProgram::new_mock()),
                SBPFVersion::V0,
                registry,
            )
            .unwrap();
            executable.set_program_id(program_id);
            executable.verify::<RequisiteVerifier>().unwrap();
            executable.jit_compile().unwrap();
            executable
        };

        let first = compile(PROGRAM_ID, 7);
        let retained = first.get_compiled_program().unwrap();
        let first_dump = dir.join(format!("jit-{pid}-{PROGRAM_ID}-1.dump"));
        let first_records = read_records(File::open(&first_dump).unwrap());
        assert!(!first_records.iter().any(|(id, _)| *id == CODE_CLOSE_ID));
        assert_eq!(
            first_records
                .iter()
                .filter(|(id, _)| *id == CODE_DEBUG_INFO_ID)
                .count(),
            3
        );
        assert_eq!(std::fs::read(&existing_dump).unwrap(), b"existing dump");
        assert_eq!(std::fs::read(&existing_source).unwrap(), b"existing source");

        first.jit_compile().unwrap();
        assert_eq!(
            read_records(File::open(&first_dump).unwrap()),
            first_records
        );
        let barrier = Barrier::new(2);
        let concurrent = thread::scope(|scope| {
            let same_program = scope.spawn(|| {
                barrier.wait();
                compile(PROGRAM_ID, 9)
            });
            let other_program = scope.spawn(|| {
                barrier.wait();
                compile(OTHER_PROGRAM_ID, 11)
            });
            (same_program.join().unwrap(), other_program.join().unwrap())
        });

        let mut dumps = BTreeMap::new();
        let mut sources = BTreeMap::new();
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("dump")
                || path == existing_dump
            {
                continue;
            }
            let stem = path.file_stem().unwrap().to_str().unwrap();
            let (program_id, value) = if stem.contains(OTHER_PROGRAM_ID) {
                (OTHER_PROGRAM_ID, 11)
            } else if stem.ends_with("-1") || stem.ends_with("-2") {
                (PROGRAM_ID, 7)
            } else {
                (PROGRAM_ID, 9)
            };
            let mut symbols = BTreeSet::new();
            let mut source_paths = BTreeSet::new();
            for (id, payload) in read_records(File::open(&path).unwrap()) {
                match id {
                    CODE_LOAD_ID => {
                        let code_len = LittleEndian::read_u64(&payload[24..]) as usize;
                        let name_end =
                            payload[40..].iter().position(|byte| *byte == 0).unwrap() + 40;
                        let name = std::str::from_utf8(&payload[40..name_end]).unwrap();
                        assert!(code_len > 0 && name_end + 1 + code_len <= payload.len());
                        symbols.insert(name.to_string());
                    }
                    CODE_DEBUG_INFO_ID => {
                        let count = LittleEndian::read_u64(&payload[8..]);
                        assert_eq!(count, 2);
                        let mut entries = &payload[16..];
                        for line in 1..=count as u32 {
                            assert_eq!(LittleEndian::read_u32(&entries[8..]), line);
                            let end =
                                entries[16..].iter().position(|byte| *byte == 0).unwrap() + 16;
                            let source_path =
                                PathBuf::from(std::str::from_utf8(&entries[16..end]).unwrap());
                            assert!(source_path
                                .file_name()
                                .unwrap()
                                .to_str()
                                .unwrap()
                                .starts_with(&format!("{stem}-")));
                            assert_ne!(source_path, existing_source);
                            source_paths.insert(source_path);
                            entries = &entries[end + 1..];
                        }
                    }
                    _ => panic!("Unexpected record before unload: {}", id),
                }
            }
            for name in [program_id, "shared", "shared/name", "shared?name"] {
                assert!(symbols.contains(name), "Missing symbol {}", name);
            }
            assert_eq!(source_paths.len(), if path == first_dump { 3 } else { 4 });
            for source_path in source_paths {
                let source = std::fs::read_to_string(&source_path).unwrap();
                assert!(source.contains(&format!("b7 00 00 00 {value:02x} 00 00 00")));
                assert_eq!(sources.insert(source_path, (path.clone(), source)), None);
            }
            dumps.insert(
                path.clone(),
                (File::open(&path).unwrap(), std::fs::read(path).unwrap()),
            );
        }
        assert_eq!(dumps.len(), 4);
        assert_eq!(sources.len(), 15);

        drop(retained);
        for (path, (_, original)) in &dumps {
            if *path == first_dump {
                assert!(!path.try_exists().unwrap());
            } else {
                assert_eq!(std::fs::read(path).unwrap(), *original);
            }
        }
        for (path, (dump_path, original)) in &sources {
            if *dump_path == first_dump {
                assert!(!path.try_exists().unwrap());
            } else {
                assert_eq!(std::fs::read_to_string(path).unwrap(), *original);
            }
        }
        drop(first);
        drop(concurrent);
        for (path, (reader, original)) in dumps {
            assert!(!path.try_exists().unwrap());
            // Readers opened before unlinking still see exactly one terminal CLOSE.
            let mut records = read_records(reader);
            assert_eq!(records.pop(), Some((CODE_CLOSE_ID, Vec::new())));
            assert_eq!(records, read_records(original.as_slice()));
        }
        for path in sources.keys() {
            assert!(!path.try_exists().unwrap());
        }

        let reloaded = compile(PROGRAM_ID, 13);
        let reloaded_dump = dir.join(format!("jit-{pid}-{PROGRAM_ID}-5.dump"));
        assert!(reloaded_dump.try_exists().unwrap());
        drop(reloaded);
        assert!(!reloaded_dump.try_exists().unwrap());
        // An unusable output directory must not make compilation fail.
        std::env::set_var("JITDUMP_DIR", &existing_dump);
        drop(compile(PROGRAM_ID, 15));
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
        assert_eq!(std::fs::read(existing_dump).unwrap(), b"existing dump");
        assert_eq!(std::fs::read(existing_source).unwrap(), b"existing source");
    }

    #[test]
    fn test_dump_cleanup_without_code() {
        let dir = std::env::temp_dir().join(format!("sbpf-jitdump-{}", rand::random::<u64>()));
        let mut writer = JitDumpWriter::new(&dir, "empty").unwrap();
        let reader = File::open(&writer.dump_path).unwrap();
        let missing_source = writer.source_path(0);
        drop(writer.create_source_file(&missing_source).unwrap());
        std::fs::remove_file(missing_source).unwrap();
        // Sources belong to the dump even before writing their contents completes.
        writer
            .create_source_file(&writer.source_path(1))
            .unwrap()
            .write_all(b"partial source")
            .unwrap();
        drop(writer);
        assert_eq!(read_records(reader), vec![(CODE_CLOSE_ID, Vec::new())]);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn test_entrypoint_aliases_share_one_code_range() {
        let mut text = vec![0x90u8; 24];
        for start in [0, 8, 16] {
            text[start..start + 4].copy_from_slice(&[0x55, 0x48, 0x89, 0xe5]);
        }
        let pc_offsets = [0, 4, 8, 12, 16, 20];
        let meta = JitCodeMeta {
            id: "my_program".to_string(),
            code_ptr: 0x1000,
            code_len: text.len(),
            sbpf_version: SBPFVersion::V0,
            symbols: vec![
                JitSymbol {
                    key: 0,
                    name: "before".to_string(),
                    pc: 0,
                },
                JitSymbol {
                    key: 1,
                    name: "my_program".to_string(),
                    pc: 2,
                },
                JitSymbol {
                    key: 2,
                    name: "my_program".to_string(),
                    pc: 2,
                },
                JitSymbol {
                    key: 3,
                    name: "after".to_string(),
                    pc: 4,
                },
            ],
            syscalls: Vec::new(),
            host_symbols: Vec::new(),
            function_entry_offsets: vec![0, u32::MAX, 8, u32::MAX, 16, u32::MAX],
        };
        let ranges = build_function_ranges(&meta, &text, &pc_offsets);
        assert_eq!(
            ranges
                .iter()
                .map(|range| (
                    range.name.as_str(),
                    range.pc_start..range.pc_end,
                    range.host_start..range.host_end,
                ))
                .collect::<Vec<_>>(),
            vec![
                ("before", 0..2, 0..8),
                ("my_program", 2..4, 8..16),
                ("after", 4..6, 16..24),
            ]
        );
        // Disassembly still needs to resolve calls through either alias key.
        let registry = build_function_registry(&meta);
        for key in [1, 2] {
            assert_eq!(
                registry.lookup_by_key(key),
                Some((b"my_program".as_slice(), 2))
            );
        }
    }
}
