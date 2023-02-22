#![cfg(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64"))]
// Copyright 2016 6WIND S.A. <quentin.monnet@6wind.com>
//
// Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or
// the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// This crate would be needed to load bytecode from a BPF-compiled object file. Since the crate
// is not used anywhere else in the library, it is deactivated: we do not want to load and compile
// it just for the tests. If you want to use it, do not forget to add the following
// dependency to your Cargo.toml file:
//
// ---
// elf = "0.0.10"
// ---
//
// extern crate elf;
// use std::path::PathBuf;

extern crate byteorder;
extern crate libc;
extern crate solana_rbpf;
extern crate test_utils;

use solana_rbpf::{
    elf::Executable,
    fuzz::fuzz,
    syscalls,
    verifier::RequisiteVerifier,
    vm::{BuiltInProgram, EbpfVm, TestContextObject, VerifiedExecutable},
};
use std::{fs::File, io::Read, sync::Arc};

// The following two examples have been compiled from C with the following command:
//
// ```bash
//  clang -O2 -emit-llvm -c <file.c> -o - | llc -march=bpf -filetype=obj -o <file.o>
// ```
//
// The C source code was the following:
//
// ```c
// #include <linux/ip.h>
// #include <linux/in.h>
// #include <linux/tcp.h>
// #include <linux/bpf.h>
//
// #define ETH_ALEN 6
// #define ETH_P_IP 0x0008 /* htons(0x0800) */
// #define TCP_HDR_LEN 20
//
// #define BLOCKED_TCP_PORT 0x9999
//
// struct eth_hdr {
//     unsigned char   h_dest[ETH_ALEN];
//     unsigned char   h_source[ETH_ALEN];
//     unsigned short  h_proto;
// };
//
// #define SEC(NAME) __attribute__((section(NAME), used))
// SEC(".classifier")
// int handle_ingress(struct __sk_buff *skb)
// {
//     void *data = (void *)(long)skb->data;
//     void *data_end = (void *)(long)skb->data_end;
//     struct eth_hdr *eth = data;
//     struct iphdr *iph = data + sizeof(*eth);
//     struct tcphdr *tcp = data + sizeof(*eth) + sizeof(*iph);
//
//     /* single length check */
//     if (data + sizeof(*eth) + sizeof(*iph) + sizeof(*tcp) > data_end)
//         return 0;
//     if (eth->h_proto != ETH_P_IP)
//         return 0;
//     if (iph->protocol != IPPROTO_TCP)
//         return 0;
//     if (tcp->source == BLOCKED_TCP_PORT || tcp->dest == BLOCKED_TCP_PORT)
//         return -1;
//     return 0;
// }
// char _license[] SEC(".license") = "GPL";
// ```
//
// This program, once compiled, can be injected into Linux kernel, with tc for instance. Sadly, we
// need to bring some modifications to the generated bytecode in order to run it: the three
// instructions with opcode 0x61 load data from a packet area as 4-byte words, where we need to
// load it as 8-bytes double words (0x79). The kernel does the same kind of translation before
// running the program, but rbpf does not implement this.
//
// In addition, the offset at which the pointer to the packet data is stored must be changed: since
// we use 8 bytes instead of 4 for the start and end addresses of the data packet, we cannot use
// the offsets produced by clang (0x4c and 0x50), the addresses would overlap. Instead we can use,
// for example, 0x40 and 0x50. See comments on the bytecode below to see the modifications.
//
// Once the bytecode has been (manually, in our case) edited, we can load the bytecode directly
// from the ELF object file. This is easy to do, but requires the addition of two crates in the
// Cargo.toml file (see comments above), so here we use just the hardcoded bytecode instructions
// instead.

#[test]
#[ignore]
fn test_fuzz_execute() {
    let mut file = File::open("tests/elfs/pass_stack_reference.so").expect("file open failed");
    let mut elf = Vec::new();
    file.read_to_end(&mut elf).unwrap();

    let mut loader = BuiltInProgram::default();
    loader
        .register_function_by_name("log", syscalls::bpf_syscall_string)
        .unwrap();
    loader
        .register_function_by_name("log_64", syscalls::bpf_syscall_u64)
        .unwrap();
    let loader = Arc::new(loader);

    println!("mangle the whole file");
    fuzz(
        &elf,
        1_000_000_000,
        100,
        0..elf.len(),
        0..255,
        |bytes: &mut [u8]| {
            if let Ok(executable) = Executable::<TestContextObject>::from_elf(bytes, loader.clone())
            {
                if let Ok(verified_executable) = VerifiedExecutable::<
                    RequisiteVerifier,
                    TestContextObject,
                >::from_executable(executable)
                {
                    let mut context_object = TestContextObject::new(1_000_000);
                    let mut vm = EbpfVm::<RequisiteVerifier, TestContextObject>::new(
                        &verified_executable,
                        &mut context_object,
                        &mut [],
                        Vec::new(),
                        None,
                    )
                    .unwrap();
                    let _ = vm.execute_program(true);
                }
            }
        },
    );
}
