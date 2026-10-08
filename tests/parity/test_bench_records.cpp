#include <lpl/bench/EnergyMeter.hpp>
#include <lpl/bench/Harness.hpp>
#include <lpl/bench/JsonRows.hpp>
#include <lpl/bench/SystemInfo.hpp>
#include <lplplugin/config.h>

#include <cctype>
#include <chrono>
#include <cstdio>
#include <expected>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

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
    expect(here.commit == LPLPLUGIN_COMMIT,
           "lpl-bench reports the commit this build was stamped with, unknown on both sides without git");
    expect(here.commit == "unknown" || isStampedCommit(here.commit),
           "a stamped commit is the short hash, with -dirty for a modified tree");
    expect(lpl::bench::machineClass(here).find(here.cpu) != std::string::npos,
           "the class of this machine names its processor");
    std::printf("  commit: %s\n  machine class: %s\n", here.commit.c_str(), lpl::bench::machineClass(here).c_str());
}

lpl::bench::Result exampleResult()
{
    lpl::bench::Result result;
    result.minNs = 91180.5;
    result.medianNs = 100300.0;
    result.meanNs = 100000.0;
    result.p99Ns = 249400.0;
    result.stddevNs = 25000.0;
    result.samples = 1373u;
    result.microjoulesPerRep = 12.5;
    return result;
}

void expectAMeasuredRowToHoldEveryFieldInOrder()
{
    expect(lpl::bench::formatJsonRow("ArenaAllocator 10k allocs", exampleResult(), exampleMachine()) ==
               R"({"schema":1,"label":"ArenaAllocator 10k allocs","median_ns":100300,"cv_percent":25,)"
               R"("min_ns":91180.5,"p99_ns":249400,"n":1373,"energy_uj_per_rep":12.5,"energy_absent_reason":null,)"
               R"("commit":"0123abc","build":"Release","compiler":"GCC 15.2.0",)"
               R"("machine_class":"Linux x86_64, Example CPU 9000, 8 logical cores, bare metal"})",
           "a measured row holds every field, in the documented order, and a null reason");
}

void expectARowWithoutEnergyToSayWhy()
{
    lpl::bench::Result withoutEnergy = exampleResult();
    withoutEnergy.microjoulesPerRep = std::unexpected{lpl::bench::EnergyAbsence::Absent};
    const std::string row = lpl::bench::formatJsonRow("ArenaAllocator 10k allocs", withoutEnergy, exampleMachine());
    expect(row.find(R"("n":1373,"energy_uj_per_rep":null,"energy_absent_reason":"absent","commit":)") !=
               std::string::npos,
           "a row without energy writes null, never zero, and the reason beside it");

    withoutEnergy.microjoulesPerRep = std::unexpected{lpl::bench::EnergyAbsence::ShortWindow};
    expect(lpl::bench::formatJsonRow("x", withoutEnergy, exampleMachine())
                   .find(R"("energy_absent_reason":"short-window")") != std::string::npos,
           "the reason is the stable name of the absence");
}

void expectStringsToBeEscaped()
{
    const std::string label = "quote \" backslash \\ newline \n tab \t control \x01 unit separator \x1f delete \x7f "
                              "N\u00b2 \xe2\x80\xa6";
    const std::string row = lpl::bench::formatJsonRow(label, exampleResult(), exampleMachine());
    expect(row.find(R"("label":"quote \" backslash \\ newline \n tab \t control \u0001 unit separator \u001f delete )"
                    "\x7f N\u00b2 \xe2\x80\xa6\",") != std::string::npos,
           "quotes, backslashes and control characters are escaped; other bytes pass as they are");
}

void expectANumberThatIsNotFiniteToBeNull()
{
    lpl::bench::Result notFinite = exampleResult();
    notFinite.medianNs = std::numeric_limits<double>::quiet_NaN();
    notFinite.p99Ns = std::numeric_limits<double>::infinity();
    const std::string row = lpl::bench::formatJsonRow("x", notFinite, exampleMachine());
    expect(row.find(R"("median_ns":null,)") != std::string::npos, "a NaN is written null, which JSON can carry");
    expect(row.find(R"("p99_ns":null,)") != std::string::npos, "an infinity is written null, which JSON can carry");

    lpl::bench::Result zeroMean = exampleResult();
    zeroMean.meanNs = 0.0;
    zeroMean.stddevNs = 0.0;
    expect(lpl::bench::formatJsonRow("x", zeroMean, exampleMachine()).find(R"("cv_percent":null,)") !=
               std::string::npos,
           "a spread over a mean of zero means nothing, and is written null rather than zero");
}

void expectTheDocumentedReasonsToBeTheWrittenOnes()
{
    std::string_view reasonMeaning;
    for (const lpl::bench::JsonRowField &field : lpl::bench::kJsonRowFields)
        if (field.key == "energy_absent_reason")
            reasonMeaning = field.meaning;
    expect(!reasonMeaning.empty(), "the fields document energy_absent_reason");

    const lpl::bench::EnergyAbsence reasons[] = {
        lpl::bench::EnergyAbsence::Denied,        lpl::bench::EnergyAbsence::Unreadable,
        lpl::bench::EnergyAbsence::Absent,        lpl::bench::EnergyAbsence::NoRepetition,
        lpl::bench::EnergyAbsence::ShortWindow,   lpl::bench::EnergyAbsence::ReadFailed,
        lpl::bench::EnergyAbsence::AmbiguousWrap,
    };
    for (const lpl::bench::EnergyAbsence reason : reasons)
        expect(reasonMeaning.find(lpl::bench::energyAbsenceName(reason)) != std::string_view::npos,
               "every reason a row can write is named where the fields are documented");
}

std::filesystem::path scratchFile(const char *scenario)
{
    return std::filesystem::temp_directory_path() /
           ("lpl-bench-rows-" + std::string{scenario} + "-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".jsonl");
}

std::vector<std::string> linesOf(const std::filesystem::path &path)
{
    std::vector<std::string> lines;
    std::ifstream input{path};
    for (std::string line; std::getline(input, line);)
        lines.push_back(line);
    return lines;
}

void expectAFileToHoldOneRowPerLabel()
{
    const std::filesystem::path path = scratchFile("file");
    {
        std::expected<lpl::bench::JsonRowFile, std::string> file =
            lpl::bench::JsonRowFile::create(path.string(), exampleMachine());
        expect(file.has_value(), "a file in a writable directory is created");
        if (!file)
            return;

        file->append("first", exampleResult());
        file->append("second", exampleResult());
        expect(!file->failure().has_value(), "two rows with their own labels are written without a failure");
        file->append("first", exampleResult());
        expect(file->failure().has_value() && file->failure()->find("'first'") != std::string::npos,
               "a second row with a label already written is refused, and the refusal names the label");
        expect(file->rowCount() == 2u, "the refused row is not counted");
    }

    const std::vector<std::string> lines = linesOf(path);
    expect(lines.size() == 2u, "the file holds one line per row written");
    if (lines.size() == 2u)
    {
        expect(lines[0] == lpl::bench::formatJsonRow("first", exampleResult(), exampleMachine()),
               "each line is the row formatJsonRow gives");
        expect(lines[1] == lpl::bench::formatJsonRow("second", exampleResult(), exampleMachine()),
               "rows keep the order they were measured in");
    }
    std::filesystem::remove(path);
}

void expectAnExistingFileToBeOverwritten()
{
    const std::filesystem::path path = scratchFile("overwrite");
    std::ofstream{path} << "a previous run\nwith two lines\n";
    {
        std::expected<lpl::bench::JsonRowFile, std::string> file =
            lpl::bench::JsonRowFile::create(path.string(), exampleMachine());
        if (file)
            file->append("only", exampleResult());
    }
    expect(linesOf(path).size() == 1u, "a file that already exists is overwritten, not appended to");
    std::filesystem::remove(path);
}

void expectAPathThatCannotBeOpenedToBeNamed()
{
    const std::filesystem::path path = scratchFile("missing-directory") / "rows.jsonl";
    const std::expected<lpl::bench::JsonRowFile, std::string> file =
        lpl::bench::JsonRowFile::create(path.string(), exampleMachine());
    expect(!file.has_value(), "a path in a missing directory cannot be opened");
    if (!file)
        expect(file.error().find(path.string()) != std::string::npos, "the refusal names the path it could not open");
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

    expectAMeasuredRowToHoldEveryFieldInOrder();
    expectARowWithoutEnergyToSayWhy();
    expectStringsToBeEscaped();
    expectANumberThatIsNotFiniteToBeNull();
    expectTheDocumentedReasonsToBeTheWrittenOnes();
    expectAFileToHoldOneRowPerLabel();
    expectAnExistingFileToBeOverwritten();
    expectAPathThatCannotBeOpenedToBeNamed();

    if (failures == 0)
        std::printf("ALL PASS (0 failures, %d checks)\n", checks);
    else
        std::printf("FAILED (%d failures, %d checks)\n", failures, checks);
    return failures == 0 ? 0 : 1;
}
