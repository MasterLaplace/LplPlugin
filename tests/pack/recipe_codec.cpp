#include <lpl/core/Types.hpp>
#include <lpl/ecology/LivingRecipe.hpp>
#include <lpl/pack/Cartridge.hpp>
#include <lpl/pack/GamePack.hpp>
#include <lpl/pack/ParityPackBlob.hpp>
#include <lpl/pack/RecipeCodec.hpp>
#include <lpl/pack/ViewerPackBlob.hpp>
#include <lpl/procgen/WorldRecipe.hpp>
#include <lpl/std/cstring.hpp>
#include <lpl/std/vector.hpp>
#include <lpl/testing/Test.hpp>

#include <cstddef>
#include <string_view>

LPL_TEST_SUITE(recipe_codec);

namespace {

/** @brief Seed of the recipes a refused image must leave in place, which no pack here carries. */
constexpr lpl::core::u32 kDefaultsSeed = 0x5EEDDEFAu;

/** @brief A word no enum of the wire format names. */
constexpr lpl::core::u32 kUnknownWord = 99u;

/** @brief Where the biome of the first scatter rule sits in a world recipe. */
constexpr lpl::core::u32 kFirstRuleBiome =
    offsetof(lpl::pack::RecipeV1, scatter) + offsetof(lpl::pack::ScatterV1, biome);

/** @brief Where the trophic level of the first species sits in a living recipe. */
constexpr lpl::core::u32 kFirstSpeciesLevel =
    offsetof(lpl::pack::LivingV1, species) + offsetof(lpl::pack::LivingSpeciesV1, level);

/**
 * @brief Recomputes the content hash of @p image, so a field written into it is the only thing
 *        wrong with it.
 */
void rehash(lpl::pmr::vector<lpl::core::u8> &image)
{
    lpl::pack::Header header{};

    lpl::pmr::memcpy(&header, image.data(), sizeof(header));
    header.contentHash = lpl::pack::hashBytes(image.data() + sizeof(header),
                                              header.totalSize - static_cast<lpl::core::u32>(sizeof(header)));
    lpl::pmr::memcpy(image.data(), &header, sizeof(header));
}

/**
 * @brief Copies the pack @p bytes and writes @p word into it, @p fieldOffset bytes into its
 *        @p section, then rehashes it.
 *
 * @return The image, or an empty one when @p bytes does not open or carries no @p section.
 */
[[nodiscard]] lpl::pmr::vector<lpl::core::u8> imageWithWord(const lpl::core::u8 *bytes, lpl::core::u32 size,
                                                            lpl::pack::SectionType section, lpl::core::u32 fieldOffset,
                                                            lpl::core::u32 word)
{
    lpl::pmr::vector<lpl::core::u8> image;
    lpl::pack::View view;
    const lpl::core::u8 *payload = nullptr;
    lpl::core::u32 payloadSize = 0u;

    image.resize(size);
    lpl::pmr::memcpy(image.data(), bytes, size);
    if (!view.open(image.data(), size) || !view.findSection(section, payload, payloadSize))
        return {};
    lpl::pmr::memcpy(image.data() + (payload - image.data()) + fieldOffset, &word, sizeof(word));
    rehash(image);
    return image;
}

/** @brief The parity pack, with @p word written into its world recipe at @p fieldOffset. */
[[nodiscard]] lpl::pmr::vector<lpl::core::u8> parityWith(lpl::core::u32 fieldOffset, lpl::core::u32 word)
{
    return imageWithWord(lpl::pack::kParityPackBytes, lpl::pack::kParityPackSize, lpl::pack::SectionType::WorldRecipe,
                         fieldOffset, word);
}

/** @brief The viewer's pack, with @p word written into its living recipe at @p fieldOffset. */
[[nodiscard]] lpl::pmr::vector<lpl::core::u8> viewerLivingWith(lpl::core::u32 fieldOffset, lpl::core::u32 word)
{
    return imageWithWord(lpl::pack::kViewerPackBytes, lpl::pack::kViewerPackSize, lpl::pack::SectionType::LivingRecipe,
                         fieldOffset, word);
}

/** @brief Loads @p image with defaults that no pack carries, so a refusal shows in the seeds. */
[[nodiscard]] lpl::pack::Cartridge load(const lpl::pmr::vector<lpl::core::u8> &image)
{
    lpl::procgen::WorldRecipe defaults = lpl::procgen::parityWorldRecipe();
    lpl::ecology::LivingRecipe defaultLiving = lpl::ecology::parityLivingRecipe();

    defaults.seed = kDefaultsSeed;
    defaultLiving.seed = kDefaultsSeed;
    return lpl::pack::loadCartridge(image.data(), static_cast<lpl::core::u32>(image.size()), nullptr, 0u, defaults,
                                    defaultLiving);
}

/** @brief Whether @p refusal names @p field. */
[[nodiscard]] bool names(const lpl::pack::WireRefusal &refusal, std::string_view field)
{
    return refusal.field != nullptr && std::string_view{refusal.field} == field;
}

/**
 * @brief Checks that the reader refuses @p image, whose @p field holds @p word, names both, and
 *        keeps the defaults it was given.
 */
void checkRefused(lpl::testing::Test &test, const lpl::pmr::vector<lpl::core::u8> &image, std::string_view field,
                  lpl::core::u32 word)
{
    lpl::pack::View view;

    if (!test.check(view.open(image.data(), static_cast<lpl::core::u32>(image.size())),
                    "the image is well formed, so the field alone can refuse it"))
        return;

    const lpl::pack::Cartridge cartridge = load(image);

    test.check(cartridge.failed, "the reader refuses the image");
    test.check(cartridge.source == lpl::pack::CartridgeSource::Defaults, "and runs nothing it carries");
    test.check(cartridge.recipe.seed == kDefaultsSeed && cartridge.living.seed == kDefaultsSeed,
               "the defaults it was given stay in place");
    test.check(names(cartridge.refusal, field), "the refusal names the field");
    test.check(cartridge.refusal.value == word, "and the value it held");
}

/** @brief Whether @p recipe comes back from the wire, decoded into @p outDecoded. */
[[nodiscard]] bool roundTrips(const lpl::procgen::WorldRecipe &recipe, lpl::procgen::WorldRecipe &outDecoded)
{
    lpl::pack::WireRefusal refusal{};

    return lpl::pack::toEngineRecipe(lpl::pack::toWireRecipe(recipe), outDecoded, refusal);
}

/** @brief Whether @p living comes back from the wire, decoded into @p outDecoded. */
[[nodiscard]] bool roundTrips(const lpl::ecology::LivingRecipe &living, lpl::ecology::LivingRecipe &outDecoded)
{
    lpl::pack::WireRefusal refusal{};

    return lpl::pack::toEngineLiving(lpl::pack::toWireLiving(living), outDecoded, refusal);
}

} // namespace

/**
 * @brief The control: a word written the same way into a field that is not an enum leaves an
 *        image the reader accepts, world and ecosystem alike, with nothing refused.
 */
LPL_TEST(a_well_formed_image_is_accepted)
{
    constexpr lpl::core::u32 kSeed = 4242u;
    const lpl::pack::Cartridge world = load(parityWith(offsetof(lpl::pack::RecipeV1, seed), kSeed));
    const lpl::pack::Cartridge living = load(viewerLivingWith(offsetof(lpl::pack::LivingV1, seed), kSeed));

    test.check(!world.failed, "the reader accepts the world");
    test.check(world.source == lpl::pack::CartridgeSource::Cartridge, "and runs the one the image carries");
    test.check(world.recipe.seed == kSeed, "with the seed written into it");
    test.check(!living.failed && living.livingFromPack, "the reader accepts the ecosystem");
    test.check(living.living.seed == kSeed, "with the seed written into it");
    test.check(world.refusal.field == nullptr && living.refusal.field == nullptr, "and refuses no field");
}

/**
 * @brief An image that fails its content hash is refused for that, and names no field: two causes
 *        of failure, told apart.
 */
LPL_TEST(a_corrupt_image_is_refused_without_naming_a_field)
{
    lpl::pmr::vector<lpl::core::u8> image = parityWith(offsetof(lpl::pack::RecipeV1, caveKind), kUnknownWord);

    if (!test.check(!image.empty(), "the image is built"))
        return;
    image[image.size() - 1u] ^= 0xFFu;

    const lpl::pack::Cartridge cartridge = load(image);

    test.check(cartridge.failed, "the reader refuses the image");
    test.check(cartridge.refusal.field == nullptr, "and names no field, since no field was read");
}

/**
 * @brief A noise kind this build does not know refuses the image, and so does one that would
 *        only fit once truncated to the enum's byte.
 */
LPL_TEST(an_unknown_noise_kind_refuses_the_image)
{
    constexpr lpl::core::u32 kTruncatesToFbm = 256u;

    checkRefused(test, parityWith(offsetof(lpl::pack::RecipeV1, noiseKind), kUnknownWord), "noiseKind", kUnknownWord);
    checkRefused(test, parityWith(offsetof(lpl::pack::RecipeV1, noiseKind), kTruncatesToFbm), "noiseKind",
                 kTruncatesToFbm);
}

/**
 * @brief A cave generator this build does not know refuses the image instead of running another.
 */
LPL_TEST(an_unknown_cave_kind_refuses_the_image)
{
    checkRefused(test, parityWith(offsetof(lpl::pack::RecipeV1, caveKind), kUnknownWord), "caveKind", kUnknownWord);
}

/**
 * @brief A province metric this build does not know refuses the image.
 */
LPL_TEST(an_unknown_province_metric_refuses_the_image)
{
    checkRefused(test, parityWith(offsetof(lpl::pack::RecipeV1, provinceMetric), kUnknownWord), "provinceMetric",
                 kUnknownWord);
}

/**
 * @brief A scatter rule naming a biome this build does not know refuses the image, and so does
 *        one naming the count of biomes, which no cell is ever classified as.
 */
LPL_TEST(an_unknown_scatter_biome_refuses_the_image)
{
    constexpr lpl::core::u32 kBiomeCount = static_cast<lpl::core::u32>(lpl::procgen::BiomeId::Count);

    checkRefused(test, parityWith(kFirstRuleBiome, kUnknownWord), "scatter.biome", kUnknownWord);
    checkRefused(test, parityWith(kFirstRuleBiome, kBiomeCount), "scatter.biome", kBiomeCount);
}

/**
 * @brief A species at a trophic level this build does not know refuses the image, its world
 *        included.
 */
LPL_TEST(an_unknown_trophic_level_refuses_the_image)
{
    checkRefused(test, viewerLivingWith(kFirstSpeciesLevel, kUnknownWord), "species.level", kUnknownWord);
}

/**
 * @brief Every value each enum of the wire names still decodes as itself: the refusal takes
 *        nothing a valid image can hold.
 */
LPL_TEST(every_known_value_still_decodes_as_itself)
{
    constexpr lpl::procgen::NoiseKind kNoises[] = {lpl::procgen::NoiseKind::Fbm, lpl::procgen::NoiseKind::Ridged,
                                                   lpl::procgen::NoiseKind::Billow};
    constexpr lpl::procgen::DistanceMetric kMetrics[] = {lpl::procgen::DistanceMetric::Euclidean,
                                                         lpl::procgen::DistanceMetric::Manhattan,
                                                         lpl::procgen::DistanceMetric::Chebyshev};
    constexpr lpl::procgen::CaveKind kCaves[] = {lpl::procgen::CaveKind::Cellular, lpl::procgen::CaveKind::Bsp,
                                                 lpl::procgen::CaveKind::Dla, lpl::procgen::CaveKind::Layered,
                                                 lpl::procgen::CaveKind::Auto};
    constexpr lpl::procgen::BiomeId kBiomes[] = {
        lpl::procgen::BiomeId::Ocean,  lpl::procgen::BiomeId::Beach,      lpl::procgen::BiomeId::Snow,
        lpl::procgen::BiomeId::Tundra, lpl::procgen::BiomeId::Taiga,      lpl::procgen::BiomeId::Rock,
        lpl::procgen::BiomeId::Desert, lpl::procgen::BiomeId::Savanna,    lpl::procgen::BiomeId::Grassland,
        lpl::procgen::BiomeId::Forest, lpl::procgen::BiomeId::Rainforest, lpl::procgen::BiomeId::Marsh,
        lpl::procgen::BiomeId::Lake};
    constexpr lpl::ecology::TrophicLevel kLevels[] = {
        lpl::ecology::TrophicLevel::Producer, lpl::ecology::TrophicLevel::Primary,
        lpl::ecology::TrophicLevel::Secondary, lpl::ecology::TrophicLevel::Apex};

    lpl::procgen::WorldRecipe recipe = lpl::procgen::parityWorldRecipe();
    lpl::procgen::WorldRecipe decoded{};
    lpl::ecology::LivingRecipe living = lpl::ecology::parityLivingRecipe();
    lpl::ecology::LivingRecipe decodedLiving{};
    bool everyNoise = true;
    bool everyMetric = true;
    bool everyCave = true;
    bool everyBiome = true;
    bool everyLevel = true;

    for (const lpl::procgen::NoiseKind kind : kNoises)
    {
        recipe.terrain.kind = kind;
        everyNoise = everyNoise && roundTrips(recipe, decoded) && decoded.terrain.kind == kind;
    }
    recipe = lpl::procgen::parityWorldRecipe();
    for (const lpl::procgen::DistanceMetric metric : kMetrics)
    {
        recipe.provinces.metric = metric;
        everyMetric = everyMetric && roundTrips(recipe, decoded) && decoded.provinces.metric == metric;
    }
    recipe = lpl::procgen::parityWorldRecipe();
    for (const lpl::procgen::CaveKind kind : kCaves)
    {
        recipe.caveKind = kind;
        everyCave = everyCave && roundTrips(recipe, decoded) && decoded.caveKind == kind;
    }
    recipe = lpl::procgen::parityWorldRecipe();
    recipe.scatterCount = 1u;
    for (const lpl::procgen::BiomeId biome : kBiomes)
    {
        recipe.scatter[0].biome = biome;
        everyBiome = everyBiome && roundTrips(recipe, decoded) && decoded.scatter[0].biome == biome;
    }
    living.speciesCount = 1u;
    for (const lpl::ecology::TrophicLevel level : kLevels)
    {
        living.species[0].params.level = level;
        everyLevel = everyLevel && roundTrips(living, decodedLiving) && decodedLiving.species[0].params.level == level;
    }

    test.check(everyNoise, "every noise kind decodes as itself");
    test.check(everyMetric, "every province metric");
    test.check(everyCave, "every cave kind, auto included");
    test.check(everyBiome, "every biome a scatter rule can name");
    test.check(everyLevel, "every trophic level");
}
