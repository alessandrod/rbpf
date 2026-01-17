use std::hash::Hasher;

use hash32::{Hasher as Hash32Hasher, Murmur3Hasher};

use crate::{elf::Executable, program::FunctionRegistry, program::SBPFVersion, vm::ContextObject};

#[cfg(feature = "jit-dump")]
mod jit_dump;
#[cfg(feature = "jit-gdb")]
mod jit_gdb;

#[allow(dead_code)]
pub struct JitSymbol {
    pub key: u32,
    pub name: String,
    pub pc: usize,
}

pub struct JitSyscallSymbol {
    pub name: String,
}

pub struct JitHostSymbol {
    pub name: String,
    pub host_start: u64,
    pub host_end: u64,
}

pub struct JitCodeMeta {
    #[allow(dead_code)]
    pub id: String,
    pub code_ptr: u64,
    #[allow(dead_code)]
    pub code_len: usize,
    pub sbpf_version: SBPFVersion,
    #[allow(dead_code)]
    pub symbols: Vec<JitSymbol>,
    pub syscalls: Vec<JitSyscallSymbol>,
    pub host_symbols: Vec<JitHostSymbol>,
    pub function_entry_offsets: Vec<u32>,
}

pub struct JitDebug {
    #[cfg(feature = "jit-dump")]
    dump: Option<jit_dump::JitDumpHook>,
    #[cfg(feature = "jit-gdb")]
    gdb: Option<jit_gdb::JitGdbHook>,
    meta: JitCodeMeta,
}

impl JitDebug {
    pub fn enabled() -> bool {
        #[cfg(feature = "jit-dump")]
        {
            if jit_dump::JitDumpHook::from_env().is_some() {
                return true;
            }
        }
        #[cfg(feature = "jit-gdb")]
        {
            return true;
        }
        false
    }

    pub fn new(meta: JitCodeMeta) -> Self {
        Self {
            #[cfg(feature = "jit-dump")]
            dump: jit_dump::JitDumpHook::from_env(),
            #[cfg(feature = "jit-gdb")]
            gdb: Some(jit_gdb::JitGdbHook::new()),
            meta,
        }
    }

    pub fn on_code_load(&mut self, text: &[u8], pc_offsets: &[u32], program: &[u8]) {
        #[cfg(feature = "jit-dump")]
        if let Some(hook) = self.dump.as_mut() {
            hook.on_code_load(&self.meta, text, pc_offsets, program);
        }
        #[cfg(feature = "jit-gdb")]
        if let Some(hook) = self.gdb.as_mut() {
            hook.on_code_load(&self.meta, text, pc_offsets, program);
        }
    }

    pub fn on_code_unload(&mut self) {
        #[cfg(feature = "jit-dump")]
        if let Some(hook) = self.dump.as_mut() {
            hook.on_code_unload();
        }
        #[cfg(feature = "jit-gdb")]
        if let Some(hook) = self.gdb.as_mut() {
            hook.on_code_unload(&self.meta);
        }
    }
}

pub fn build_jit_code_meta<C: ContextObject>(
    executable: &Executable<C>,
    program: &[u8],
    code_ptr: *const u8,
    code_len: usize,
    host_symbols: Vec<JitHostSymbol>,
    function_entry_offsets: Vec<u32>,
) -> JitCodeMeta {
    let id = if let Some(program_id) = executable.get_program_id() {
        program_id.to_string()
    } else {
        let mut hasher = Murmur3Hasher::default();
        hasher.write(program);
        let hash = hasher.finish32();
        format!("sbpf_{hash:08x}")
    };

    let symbols = collect_symbols(executable, &id);
    let syscalls = collect_syscalls(executable.get_loader().get_function_registry());

    JitCodeMeta {
        id,
        code_ptr: code_ptr as u64,
        code_len,
        sbpf_version: executable.get_sbpf_version(),
        symbols,
        syscalls,
        host_symbols,
        function_entry_offsets,
    }
}

fn collect_symbols<C: ContextObject>(executable: &Executable<C>, id: &str) -> Vec<JitSymbol> {
    let entry_pc = executable.get_entrypoint_instruction_offset();
    let entry_name = executable.get_program_id().map(str::as_bytes);
    let mut symbols: Vec<_> = executable
        .get_function_registry()
        .iter()
        .filter_map(|(key, (name, pc))| {
            let name = if pc == entry_pc {
                entry_name.unwrap_or(name)
            } else {
                name
            };
            if name.is_empty() {
                return None;
            }
            Some(JitSymbol {
                key,
                name: String::from_utf8_lossy(name).to_string(),
                pc,
            })
        })
        .collect();
    // Stripped ELFs may have no entrypoint symbol. Its address is still known.
    if !symbols.iter().any(|symbol| symbol.pc == entry_pc) {
        symbols.push(JitSymbol {
            key: if executable.get_sbpf_version().static_syscalls() {
                entry_pc as u32
            } else {
                crate::ebpf::hash_symbol_name(b"entrypoint")
            },
            name: id.to_string(),
            pc: entry_pc,
        });
    }
    symbols
}

fn collect_syscalls<C: ContextObject>(
    registry: &FunctionRegistry<(
        crate::program::BuiltinFunction<C>,
        crate::program::BuiltinCodegen<C>,
    )>,
) -> Vec<JitSyscallSymbol> {
    registry
        .iter()
        .filter_map(|(_key, (name, _value))| {
            if name.is_empty() {
                return None;
            }
            Some(JitSyscallSymbol {
                name: String::from_utf8_lossy(name).to_string(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ebpf, elf_parser::types::Elf64Ehdr, program::BuiltinProgram,
        static_analysis::DummyContextObject, vm::Config,
    };
    use std::sync::Arc;

    #[test]
    fn test_program_id_names_entrypoint_aliases() {
        let program = [0u8; 4 * ebpf::INSN_SIZE];
        let mut registry = FunctionRegistry::default();
        registry
            .register_function(0, b"helper".to_vec(), 0)
            .unwrap();
        registry.register_function(1, b"alias".to_vec(), 2).unwrap();
        registry
            .register_function(2, b"entrypoint".to_vec(), 2)
            .unwrap();
        registry.register_function(3, Vec::new(), 2).unwrap();
        let mut executable = Executable::<DummyContextObject>::from_text_bytes(
            &program,
            Arc::new(BuiltinProgram::new_mock()),
            SBPFVersion::V0,
            registry,
        )
        .unwrap();

        for program_id in [None, Some("my_program")] {
            if let Some(id) = program_id {
                executable.set_program_id(id);
            }
            let meta = build_jit_code_meta(
                &executable,
                &program,
                std::ptr::null(),
                0,
                Vec::new(),
                Vec::new(),
            );
            let symbols: Vec<_> = meta
                .symbols
                .iter()
                .map(|symbol| (symbol.key, symbol.name.as_str(), symbol.pc))
                .collect();
            if program_id.is_some() {
                assert_eq!(meta.id, "my_program");
                assert_eq!(
                    symbols,
                    vec![
                        (0, "helper", 0),
                        (1, "my_program", 2),
                        (2, "my_program", 2),
                        (3, "my_program", 2),
                    ]
                );
            } else {
                assert_eq!(
                    symbols,
                    vec![(0, "helper", 0), (1, "alias", 2), (2, "entrypoint", 2)]
                );
            }
        }
        assert_eq!(
            executable.get_function_registry().lookup_by_key(1),
            Some((b"alias".as_slice(), 2))
        );
        assert_eq!(
            executable.get_function_registry().lookup_by_key(2),
            Some((b"entrypoint".as_slice(), 2))
        );
    }

    #[test]
    fn test_program_id_with_stripped_elf() {
        let mut elf = include_bytes!("../../tests/elfs/relative_call.so").to_vec();
        // Remove the section table, leaving the program headers and entrypoint intact.
        for (offset, size) in [
            (std::mem::offset_of!(Elf64Ehdr, e_shoff), 8),
            (std::mem::offset_of!(Elf64Ehdr, e_shnum), 2),
            (std::mem::offset_of!(Elf64Ehdr, e_shstrndx), 2),
        ] {
            elf[offset..offset + size].fill(0);
        }
        let mut executable = Executable::<DummyContextObject>::from_elf(
            &elf,
            Arc::new(BuiltinProgram::new_loader(Config {
                enable_symbol_and_section_labels: true,
                ..Config::default()
            })),
        )
        .unwrap();
        assert_eq!(executable.get_function_registry().iter().count(), 0);
        let entry_pc = executable.get_entrypoint_instruction_offset();
        assert_ne!(entry_pc, 0);

        for program_id in [None, Some("my_program")] {
            if let Some(id) = program_id {
                executable.set_program_id(id);
            }
            let meta = build_jit_code_meta(
                &executable,
                executable.get_text_bytes().1,
                std::ptr::null(),
                0,
                Vec::new(),
                Vec::new(),
            );
            assert_eq!(meta.symbols.len(), 1);
            let symbol = &meta.symbols[0];
            assert_eq!(symbol.pc, entry_pc);
            assert_eq!(symbol.key, entry_pc as u32);
            assert_eq!(symbol.name, meta.id);
            if let Some(id) = program_id {
                assert_eq!(symbol.name, id);
            } else {
                assert!(symbol.name.starts_with("sbpf_"));
            }
        }
    }
}
