#include <lpl/bench/SystemInfo.hpp>

#include <cctype>
#include <cstdio>
#include <string>
#include <string_view>

namespace {

int checks = 0;
int failures = 0;

void expect(bool condition, const char *what)
{
    ++checks;
    if (condition)
        return;
    ++failures;
    std::printf("  FAIL  %s\n", what);
}

lpl::bench::SystemInfo exampleMachine()
{
    lpl::bench::SystemInfo info;
    info.os = "Linux (6.6.114.1-microsoft-standard-WSL2) x86_64";
    info.platform = "Linux x86_64";
    info.cpu = "Example CPU 9000";
    info.logicalCores = 8u;
    info.hypervisor = "none";
    info.ramBytes = 16ull << 30u;
    info.compiler = "GCC 15.2.0";
    info.buildConfig = "Release";
    info.cpuGovernor = "performance";
    info.commit = "0123abc";
    return info;
}

void expectTheMachineClassToNameWhatGroupsRuns()
{
    lpl::bench::SystemInfo bareMetal = exampleMachine();
    expect(lpl::bench::machineClass(bareMetal) == "Linux x86_64, Example CPU 9000, 8 logical cores, bare metal",
           "a machine class names the platform, the processor, its logical cores and bare metal");

    lpl::bench::SystemInfo guest = exampleMachine();
    guest.hypervisor = "KVMKVMKVM";
    expect(lpl::bench::machineClass(guest) == "Linux x86_64, Example CPU 9000, 8 logical cores, hypervisor KVMKVMKVM",
           "a machine class names the hypervisor a guest runs under");

    lpl::bench::SystemInfo undetected = exampleMachine();
    undetected.hypervisor = "unknown";
    expect(lpl::bench::machineClass(undetected) ==
               "Linux x86_64, Example CPU 9000, 8 logical cores, hypervisor unknown",
           "a machine class says when it could not tell whether a hypervisor runs");
}

void expectTheMachineClassToIgnoreWhatChangesBetweenRunsOfOneMachine()
{
    const std::string reference = lpl::bench::machineClass(exampleMachine());

    lpl::bench::SystemInfo other = exampleMachine();
    other.os = "Linux (6.6.200-microsoft-standard-WSL2) x86_64";
    other.ramBytes = 32ull << 30u;
    other.compiler = "Clang 19.1.0";
    other.buildConfig = "Debug";
    other.cpuGovernor = "powersave";
    other.commit = "4567def-dirty";
    expect(lpl::bench::machineClass(other) == reference,
           "the kernel release, the RAM, the compiler, the build, the governor and the commit leave the class "
           "alone");
}

void expectTheMachineClassToSeparateDifferentMachines()
{
    const std::string reference = lpl::bench::machineClass(exampleMachine());

    lpl::bench::SystemInfo moreCores = exampleMachine();
    moreCores.logicalCores = 16u;
    expect(lpl::bench::machineClass(moreCores) != reference, "another core count is another class");

    lpl::bench::SystemInfo otherProcessor = exampleMachine();
    otherProcessor.cpu = "Example CPU 9100";
    expect(lpl::bench::machineClass(otherProcessor) != reference, "another processor is another class");

    lpl::bench::SystemInfo underHypervisor = exampleMachine();
    underHypervisor.hypervisor = "Microsoft Hv";
    expect(lpl::bench::machineClass(underHypervisor) != reference, "a guest is not its bare-metal host");

    lpl::bench::SystemInfo otherPlatform = exampleMachine();
    otherPlatform.platform = "Windows x86_64";
    expect(lpl::bench::machineClass(otherPlatform) != reference, "another operating system is another class");
}

bool isStampedCommit(std::string_view commit)
{
    constexpr std::string_view kDirtySuffix = "-dirty";
    if (commit.ends_with(kDirtySuffix))
        commit.remove_suffix(kDirtySuffix.size());
    if (commit.size() != 7u)
        return false;
    for (const char digit : commit)
        if (std::isxdigit(static_cast<unsigned char>(digit)) == 0 || std::isupper(static_cast<unsigned char>(digit)))
            return false;
    return true;
}

void expectThisBuildToKnowItsCommit()
{
    const lpl::bench::SystemInfo here = lpl::bench::collectSystemInfo();
    expect(isStampedCommit(here.commit), "the build stamps the short commit, with -dirty for a modified tree");
    expect(lpl::bench::machineClass(here).find(here.cpu) != std::string::npos,
           "the class of this machine names its processor");
    std::printf("  commit: %s\n  machine class: %s\n", here.commit.c_str(), lpl::bench::machineClass(here).c_str());
}

} // namespace

int main()
{
    expect(!isStampedCommit("unknown"), "an unstamped build is not taken for a commit");
    expect(isStampedCommit("0123abc-dirty"), "a dirty commit is still a commit");

    expectTheMachineClassToNameWhatGroupsRuns();
    expectTheMachineClassToIgnoreWhatChangesBetweenRunsOfOneMachine();
    expectTheMachineClassToSeparateDifferentMachines();
    expectThisBuildToKnowItsCommit();

    if (failures == 0)
        std::printf("ALL PASS (0 failures, %d checks)\n", checks);
    else
        std::printf("FAILED (%d failures, %d checks)\n", failures, checks);
    return failures == 0 ? 0 : 1;
}
