#include <lpl/core/Types.hpp>
#include <lpl/history/Fold.hpp>
#include <lpl/net/protocol/BitError.hpp>
#include <lpl/net/protocol/BitReader.hpp>
#include <lpl/net/protocol/BitWriter.hpp>
#include <lpl/net/protocol/Quantization.hpp>
#include <lpl/testing/Test.hpp>

#include <array>
#include <bit>
#include <expected>
#include <limits>
#include <span>

LPL_TEST_SUITE(bit_stream);

namespace {

/** What a destination held before a writer touched it: none of it may show in what the writer wrote. */
constexpr lpl::core::u8 kStaleByte = 0xA5u;

/** A word whose cut at each width differs from the cuts at the widths around it. */
constexpr lpl::core::u32 kWidthPattern = 0xB5E3D2A7u;

/** What a refused read of a step stands for: no step a check here expects. */
constexpr lpl::core::u32 kNoStep = 0xDEADBEEFu;

/** Entities of the snapshot the cost of writing and reading one is counted on. */
constexpr lpl::core::u32 kSnapshotEntities = 64u;

/** Input frames written back to back, so that each starts at another bit of its byte. */
constexpr lpl::core::u32 kInputFrames = 64u;

/** Keys an input frame carries, as InputSendSystem writes them. */
constexpr lpl::core::u32 kKeysPerFrame = 4u;

/**
 * @brief How a field of the fixed message is written and read.
 */
enum class FieldKind : lpl::core::u8 {
    kBits = 1,  /**< writeBits() of the low bits of a word. */
    kBool,      /**< writeBool(). */
    kU8,        /**< writeU8(). */
    kU16,       /**< writeU16(). */
    kU32,       /**< writeU32(). */
    kI32,       /**< writeI32() of a word's two's complement bits. */
    kFloat,     /**< writeFloat() of a word's IEEE 754 bits. */
    kVarint,    /**< writeVarint(). */
    kQuantized, /**< writeQuantizedFloat(), read back as its raw step. */
    kAngle,     /**< writeAngle(), read back as its raw step. */
};

/**
 * @struct Field
 * @brief One field of the fixed message, and the word it must read back as.
 */
struct Field {
    FieldKind kind{FieldKind::kBits}; /**< How it is written and read. */
    lpl::core::u32 word{0u};          /**< The word written, or the step a quantized field must take. */
    lpl::core::u32 bits{0u};          /**< Width of a kBits field or of a step. */
    float real{0.0f};                 /**< The value a quantized field or an angle writes. */
    float minimum{0.0f};              /**< Lower bound of a quantized field. */
    float maximum{0.0f};              /**< Upper bound of a quantized field. */
};

/** Fields of the fixed message after its 32 kBits fields, one per width. */
constexpr lpl::core::u32 kFixedFields = 35u;

/** Fields of the fixed message. */
constexpr lpl::core::u32 kMessageFields = 32u + kFixedFields;

/**
 * @brief The fixed message: every width from 1 to 32 bits, then every other kind of field at its edges.
 *
 * @details The 32 widths written one after the other start each field at another bit of its byte. A
 *          quantized field carries the step it must take, worked out by hand from the range and not
 *          read from the code: 500 over [0, 1000] on 16 bits is 0.5 * 65535 + 0.5 = 32768, 250 on 10
 *          bits is 0.25 * 1023 + 0.5 = 256.25, an angle of 0.5 rad on 10 bits is 0.0795775 * 1023 +
 *          0.5 = 81.9 and one of -0.5 rad wraps to 5.7831855, 0.920423 * 1023 + 0.5 = 942.1. NaN takes
 *          the bottom step, the top of a range the top one on 32 bits too, and an angle of 1e11 rad the
 *          top step, its turns being too many to count.
 */
[[nodiscard]] constexpr std::array<Field, kMessageFields> fixedMessage() noexcept
{
    constexpr std::array<Field, kFixedFields> kEdges{
        {
         {FieldKind::kBool, 1u},
         {FieldKind::kBool, 0u},
         {FieldKind::kU8, 0x00u},
         {FieldKind::kU8, 0xFFu},
         {FieldKind::kU16, 0xBEEFu},
         {FieldKind::kU32, 0xDEADBEEFu},
         {FieldKind::kI32, 0xFFFFFFFFu},
         {FieldKind::kI32, 0x80000000u},
         {FieldKind::kFloat, 0x80000000u},
         {FieldKind::kFloat, 0x7F800000u},
         {FieldKind::kFloat, 0x7FC00001u},
         {FieldKind::kFloat, 0x00000001u},
         {FieldKind::kFloat, 0x3FC00000u},
         {FieldKind::kVarint, 0u},
         {FieldKind::kVarint, 1u},
         {FieldKind::kVarint, 127u},
         {FieldKind::kVarint, 128u},
         {FieldKind::kVarint, 300u},
         {FieldKind::kVarint, 16383u},
         {FieldKind::kVarint, 16384u},
         {FieldKind::kVarint, 0xFFFFFFFFu},
         {FieldKind::kQuantized, 0u, 16u, 0.0f, 0.0f, 1000.0f},
         {FieldKind::kQuantized, 65535u, 16u, 1000.0f, 0.0f, 1000.0f},
         {FieldKind::kQuantized, 0u, 16u, -50.0f, 0.0f, 1000.0f},
         {FieldKind::kQuantized, 65535u, 16u, 5000.0f, 0.0f, 1000.0f},
         {FieldKind::kQuantized, 32768u, 16u, 500.0f, 0.0f, 1000.0f},
         {FieldKind::kQuantized, 256u, 10u, 250.0f, 0.0f, 1000.0f},
         {FieldKind::kQuantized, 0x40000000u, 32u, 0.25f, 0.0f, 1.0f},
         {FieldKind::kAngle, 81u, 10u, 0.5f},
         {FieldKind::kAngle, 81u, 10u, 0.5f + lpl::net::protocol::kTwoPi},
         {FieldKind::kAngle, 942u, 10u, -0.5f},
         {FieldKind::kQuantized, 0u, 16u, std::numeric_limits<float>::quiet_NaN(), 0.0f, 1000.0f},
         {FieldKind::kQuantized, 0xFFFFFFFFu, 32u, 1.0f, 0.0f, 1.0f},
         {FieldKind::kAngle, 1023u, 10u, 1.0e11f},
         {FieldKind::kAngle, 0u, 10u, std::numeric_limits<float>::quiet_NaN()},
         }
    };
    std::array<Field, kMessageFields> message{};

    for (lpl::core::u32 width = 1u; width <= 32u; ++width)
        message[width - 1u] = Field{FieldKind::kBits, kWidthPattern, width};
    for (lpl::core::u32 index = 0u; index < kFixedFields; ++index)
        message[32u + index] = kEdges[index];
    return message;
}

/** The fixed message, the one table the host and ring 0 both write. */
constexpr std::array<Field, kMessageFields> kMessage = fixedMessage();

static_assert(kMessageFields > 32u, "the message holds every width, then the edges of every other kind");

/**
 * @brief The low @p bits bits of a word, worked out apart from the code under test.
 */
[[nodiscard]] constexpr lpl::core::u32 lowBits(lpl::core::u32 word, lpl::core::u32 bits) noexcept
{
    return bits >= 32u ? word : (word & ((1u << bits) - 1u));
}

/**
 * @brief The word @p field must read back as.
 */
[[nodiscard]] constexpr lpl::core::u32 expectedWord(const Field &field) noexcept
{
    if (field.kind == FieldKind::kBits)
        return lowBits(field.word, field.bits);
    if (field.kind == FieldKind::kBool)
        return field.word != 0u ? 1u : 0u;
    return field.word;
}

void writeField(lpl::net::protocol::BitWriter &writer, const Field &field) noexcept
{
    switch (field.kind)
    {
    case FieldKind::kBits: writer.writeBits(field.word, field.bits); return;
    case FieldKind::kBool: writer.writeBool(field.word != 0u); return;
    case FieldKind::kU8: writer.writeU8(static_cast<lpl::core::u8>(field.word)); return;
    case FieldKind::kU16: writer.writeU16(static_cast<lpl::core::u16>(field.word)); return;
    case FieldKind::kU32: writer.writeU32(field.word); return;
    case FieldKind::kI32: writer.writeI32(std::bit_cast<lpl::core::i32>(field.word)); return;
    case FieldKind::kFloat: writer.writeFloat(std::bit_cast<float>(field.word)); return;
    case FieldKind::kVarint: writer.writeVarint(field.word); return;
    case FieldKind::kQuantized:
        writer.writeQuantizedFloat(field.real, field.minimum, field.maximum, field.bits);
        return;
    case FieldKind::kAngle: writer.writeAngle(field.real, field.bits); return;
    }
}

[[nodiscard]] std::expected<lpl::core::u32, lpl::net::protocol::BitError>
readField(lpl::net::protocol::BitReader &reader, const Field &field) noexcept
{
    switch (field.kind)
    {
    case FieldKind::kBits: return reader.readBits(field.bits);
    case FieldKind::kBool: return reader.readBool().transform([](bool bit) { return bit ? 1u : 0u; });
    case FieldKind::kU8: return reader.readU8().transform([](lpl::core::u8 value) { return lpl::core::u32{value}; });
    case FieldKind::kU16: return reader.readU16().transform([](lpl::core::u16 value) { return lpl::core::u32{value}; });
    case FieldKind::kU32: return reader.readU32();
    case FieldKind::kI32:
        return reader.readI32().transform([](lpl::core::i32 value) { return std::bit_cast<lpl::core::u32>(value); });
    case FieldKind::kFloat:
        return reader.readFloat().transform([](float value) { return std::bit_cast<lpl::core::u32>(value); });
    case FieldKind::kVarint: return reader.readVarint();
    case FieldKind::kQuantized:
    case FieldKind::kAngle: return reader.readBits(field.bits);
    }
    return std::unexpected(lpl::net::protocol::BitError::kInvalidWidth);
}

/**
 * @brief @p value, which the compiler cannot see through: the arithmetic the test exercises runs on the
 *        target, as it does for a value from the simulation, instead of being folded at compile time.
 */
[[gnu::noipa]] float atRunTime(float value) noexcept { return value; }

/**
 * @brief The FNV-1a fold of @p bytes, one byte a step.
 */
[[nodiscard]] lpl::core::u32 foldBytes(std::span<const lpl::core::byte> bytes) noexcept
{
    lpl::core::u32 signature = lpl::history::kFoldOffsetBasis;

    for (const lpl::core::byte octet : bytes)
        signature = lpl::history::foldWord(signature, static_cast<lpl::core::u32>(octet));
    return signature;
}

/**
 * @struct MessageRead
 * @brief What reading the fixed message back found.
 */
struct MessageRead {
    lpl::core::u32 signature{lpl::history::kFoldOffsetBasis}; /**< Fold of every word read. */
    lpl::core::u32 mismatches{0u};                            /**< Fields that read back another word. */
    lpl::core::u32 refusals{0u};                              /**< Fields the reader refused. */
};

[[nodiscard]] MessageRead readMessage(std::span<const lpl::core::byte> bytes, lpl::core::u32 bitCount) noexcept
{
    lpl::net::protocol::BitReader reader{bytes, bitCount};
    MessageRead read{};

    for (const Field &field : kMessage)
    {
        const auto word = readField(reader, field);
        if (!word.has_value())
        {
            ++read.refusals;
            continue;
        }
        read.signature = lpl::history::foldWord(read.signature, *word);
        if (*word != expectedWord(field))
            ++read.mismatches;
    }
    return read;
}

/**
 * @struct SnapshotEntity
 * @brief An entity as the full snapshot carries it: an id, a position, a size and hit points.
 */
struct SnapshotEntity {
    lpl::core::u32 id{0u};           /**< Entity id. */
    std::array<float, 3> position{}; /**< Position, x y z. */
    std::array<float, 3> size{};     /**< Size, x y z. */
    lpl::core::i32 health{0};        /**< Hit points. */
};

/**
 * @brief A deterministic entity, different for every @p index.
 */
[[nodiscard]] SnapshotEntity makeEntity(lpl::core::u32 index) noexcept
{
    const float base = static_cast<float>(index);

    return SnapshotEntity{
        index * 2654435761u,
        {base * 1.5f, -base,       base * 0.25f},
        {1.0f,        2.0f + base, 0.5f        },
        static_cast<lpl::core::i32>(100u - index)
    };
}

[[nodiscard]] bool sameEntity(const SnapshotEntity &lhs, const SnapshotEntity &rhs) noexcept
{
    bool same = lhs.id == rhs.id && lhs.health == rhs.health;

    for (lpl::core::u32 axis = 0u; axis < 3u; ++axis)
        same = same &&
               std::bit_cast<lpl::core::u32>(lhs.position[axis]) == std::bit_cast<lpl::core::u32>(rhs.position[axis]) &&
               std::bit_cast<lpl::core::u32>(lhs.size[axis]) == std::bit_cast<lpl::core::u32>(rhs.size[axis]);
    return same;
}

/**
 * @brief Writes one entity of a snapshot: the window `tools/instructions.sh --inside` counts.
 */
[[gnu::noipa]] void writeSnapshotEntity(lpl::net::protocol::BitWriter &writer, const SnapshotEntity &entity) noexcept
{
    writer.writeU32(entity.id);
    for (const float axis : entity.position)
        writer.writeFloat(axis);
    for (const float axis : entity.size)
        writer.writeFloat(axis);
    writer.writeI32(entity.health);
}

/**
 * @brief Reads one entity of a snapshot, nothing when a field is refused.
 */
[[gnu::noipa]] bool readSnapshotEntity(lpl::net::protocol::BitReader &reader, SnapshotEntity &entity) noexcept
{
    const auto id = reader.readU32();
    if (!id.has_value())
        return false;
    entity.id = *id;
    for (float &axis : entity.position)
    {
        const auto value = reader.readFloat();
        if (!value.has_value())
            return false;
        axis = *value;
    }
    for (float &axis : entity.size)
    {
        const auto value = reader.readFloat();
        if (!value.has_value())
            return false;
        axis = *value;
    }
    const auto health = reader.readI32();
    if (!health.has_value())
        return false;
    entity.health = *health;
    return true;
}

/**
 * @struct InputFrame
 * @brief An input frame as InputSendSystem writes it: an entity, its keys, no axis, the neural
 *        values and a blink.
 */
struct InputFrame {
    lpl::core::u32 entity{0u};                        /**< Entity the inputs drive. */
    std::array<lpl::core::u16, kKeysPerFrame> keys{}; /**< Key codes. */
    std::array<bool, kKeysPerFrame> pressed{};        /**< Whether each key is down. */
    std::array<float, 3> neural{};                    /**< Alpha, beta and concentration. */
    bool blink{false};                                /**< Whether the user blinked. */
};

[[nodiscard]] InputFrame makeFrame(lpl::core::u32 index) noexcept
{
    InputFrame frame{};

    frame.entity = index + 1u;
    for (lpl::core::u32 key = 0u; key < kKeysPerFrame; ++key)
    {
        frame.keys[key] = static_cast<lpl::core::u16>('W' + key + index);
        frame.pressed[key] = ((index >> key) & 1u) != 0u;
    }
    frame.neural = {0.25f * static_cast<float>(index), 0.5f, -0.75f};
    frame.blink = (index & 1u) != 0u;
    return frame;
}

[[nodiscard]] bool sameFrame(const InputFrame &lhs, const InputFrame &rhs) noexcept
{
    bool same = lhs.entity == rhs.entity && lhs.blink == rhs.blink;

    for (lpl::core::u32 key = 0u; key < kKeysPerFrame; ++key)
        same = same && lhs.keys[key] == rhs.keys[key] && lhs.pressed[key] == rhs.pressed[key];
    for (lpl::core::u32 axis = 0u; axis < 3u; ++axis)
        same =
            same && std::bit_cast<lpl::core::u32>(lhs.neural[axis]) == std::bit_cast<lpl::core::u32>(rhs.neural[axis]);
    return same;
}

/**
 * @brief Writes one input frame, 221 bits: the window `tools/instructions.sh --inside` counts.
 */
[[gnu::noipa]] void writeInputFrame(lpl::net::protocol::BitWriter &writer, const InputFrame &frame) noexcept
{
    writer.writeU32(frame.entity);
    writer.writeU16(static_cast<lpl::core::u16>(kKeysPerFrame));
    for (lpl::core::u32 key = 0u; key < kKeysPerFrame; ++key)
    {
        writer.writeU16(frame.keys[key]);
        writer.writeBool(frame.pressed[key]);
    }
    writer.writeU8(0u);
    for (const float value : frame.neural)
        writer.writeFloat(value);
    writer.writeBool(frame.blink);
}

/**
 * @brief Reads one input frame, nothing when a field is refused or the frame is not one.
 */
[[gnu::noipa]] bool readInputFrame(lpl::net::protocol::BitReader &reader, InputFrame &frame) noexcept
{
    const auto entity = reader.readU32();
    const auto keyCount = reader.readU16();
    if (!entity.has_value() || !keyCount.has_value() || *keyCount != kKeysPerFrame)
        return false;
    frame.entity = *entity;
    for (lpl::core::u32 key = 0u; key < kKeysPerFrame; ++key)
    {
        const auto code = reader.readU16();
        const auto pressed = reader.readBool();
        if (!code.has_value() || !pressed.has_value())
            return false;
        frame.keys[key] = *code;
        frame.pressed[key] = *pressed;
    }
    const auto axes = reader.readU8();
    if (!axes.has_value() || *axes != 0u)
        return false;
    for (float &value : frame.neural)
    {
        const auto read = reader.readFloat();
        if (!read.has_value())
            return false;
        value = *read;
    }
    const auto blink = reader.readBool();
    if (!blink.has_value())
        return false;
    frame.blink = *blink;
    return true;
}

} // namespace

LPL_TEST(the_first_bit_written_is_the_top_bit_of_the_first_byte)
{
    constexpr std::array<lpl::core::u8, 17> kExpected{0xBFu, 0xE4u, 0x12u, 0x34u, 0x3Fu, 0x80u, 0x00u, 0x00u, 0xACu,
                                                      0x02u, 0xFFu, 0xFFu, 0xFFu, 0xFEu, 0xD5u, 0xE6u, 0xFFu};
    std::array<lpl::core::byte, 32> wire{};
    wire.fill(lpl::core::byte{kStaleByte});
    lpl::net::protocol::BitWriter writer{wire};

    writer.writeBits(0b101u, 3u);
    writer.writeU8(0xFFu);
    writer.writeBits(0u, 2u);
    writer.writeBool(true);
    writer.writeBits(0u, 2u);
    writer.writeU16(0x1234u);
    writer.writeFloat(1.0f);
    writer.writeVarint(300u);
    writer.writeI32(-2);
    writer.writeBool(true);
    writer.writeU16(0xABCDu);
    writer.writeBits(0x7Fu, 7u);

    const std::span<const lpl::core::byte> bytes = writer.bytes();
    lpl::core::u32 differing = 0u;
    for (lpl::core::u32 index = 0u; index < kExpected.size() && index < bytes.size(); ++index)
        differing += static_cast<lpl::core::u8>(bytes[index]) != kExpected[index] ? 1u : 0u;

    test.check(!writer.failure().has_value(), "every field of the layout fits");
    test.check(writer.bitsWritten() == 136u, "the layout is 136 bits long");
    test.check(bytes.size() == kExpected.size(), "the layout covers 17 bytes");
    test.check(differing == 0u, "each byte is the one worked out by hand, top bit first, values big-endian");
    test.measure("differing_bytes", differing);
    test.measureHexadecimal("layout_signature", foldBytes(bytes));
}

LPL_TEST(a_message_of_every_field_kind_comes_back_word_for_word)
{
    std::array<lpl::core::byte, 512> wire{};
    wire.fill(lpl::core::byte{kStaleByte});
    lpl::net::protocol::BitWriter writer{wire};

    for (const Field &field : kMessage)
        writeField(writer, field);

    const std::span<const lpl::core::byte> bytes = writer.bytes();
    const lpl::core::u32 paddingBits = (8u - (writer.bitsWritten() & 7u)) & 7u;
    const lpl::core::u32 padding =
        bytes.empty() ? 0u : static_cast<lpl::core::u32>(bytes.back()) & ((1u << paddingBits) - 1u);
    const MessageRead read = readMessage(bytes, writer.bitsWritten());

    test.check(!writer.failure().has_value(), "every field of the message fits and is written");
    test.check(padding == 0u, "the bits after the last field are zero, whatever the destination held");
    test.check(read.refusals == 0u, "the reader takes every field the writer wrote");
    test.check(read.mismatches == 0u, "every field reads back as the word it was written from");
    test.measure("fields", kMessageFields);
    test.measure("bits_written", writer.bitsWritten());
    test.measure("mismatches", read.mismatches);
    test.measureHexadecimal("wire_signature", foldBytes(bytes));
    test.measureHexadecimal("read_signature", read.signature);
}

LPL_TEST(one_flipped_bit_changes_what_the_message_reads)
{
    std::array<lpl::core::byte, 512> wire{};
    lpl::net::protocol::BitWriter writer{wire};

    for (const Field &field : kMessage)
        writeField(writer, field);

    const lpl::core::u32 bitCount = writer.bitsWritten();
    const MessageRead clean = readMessage(writer.bytes(), bitCount);
    const lpl::core::u32 flipped = bitCount / 2u;
    wire[flipped >> 3u] ^= lpl::core::byte{static_cast<lpl::core::u8>(0x80u >> (flipped & 7u))};
    const MessageRead corrupted = readMessage(writer.bytes(), bitCount);

    test.check(clean.mismatches == 0u, "the clean message reads back whole");
    test.check(corrupted.signature != clean.signature, "a flipped bit changes the fold of what is read");
    test.check(corrupted.mismatches + corrupted.refusals > 0u, "a flipped bit changes at least one field");
    test.measure("flipped_bit", flipped);
    test.measureHexadecimal("corrupted_signature", corrupted.signature);
}

LPL_TEST(a_read_past_the_source_is_refused_and_moves_nothing)
{
    constexpr std::array<lpl::core::byte, 3> kSource{lpl::core::byte{0x12u}, lpl::core::byte{0x34u},
                                                     lpl::core::byte{0x56u}};
    lpl::net::protocol::BitReader reader{kSource, 24u};
    std::array<lpl::core::byte, 2> untouched{lpl::core::byte{kStaleByte}, lpl::core::byte{kStaleByte}};

    const auto tooWide = reader.readU32();
    test.check(!tooWide.has_value() && tooWide.error() == lpl::net::protocol::BitError::kUnderflow,
               "32 bits from a 24-bit source are refused as an underflow");
    test.check(reader.bitsRead() == 0u, "a refused read leaves the cursor where it was");
    test.check(reader.readU16().value_or(0u) == 0x1234u, "the bits a refused read left are still read in order");
    test.check(!reader.readU16().has_value(), "16 bits from the last 8 are refused");
    test.check(!reader.readBytes(untouched).has_value() && untouched[0] == lpl::core::byte{kStaleByte},
               "bytes past the end are refused, and the destination is left as it was");
    test.check(reader.readU8().value_or(0u) == 0x56u, "the last byte is still read whole");
    test.check(!reader.readBool().has_value() && reader.bitsRemaining() == 0u, "nothing is read past the last bit");
}

LPL_TEST(a_varint_longer_than_thirty_two_bits_is_refused_whole)
{
    constexpr std::array<lpl::core::byte, 6> kSixBytes{lpl::core::byte{0xFFu}, lpl::core::byte{0xFFu},
                                                       lpl::core::byte{0xFFu}, lpl::core::byte{0xFFu},
                                                       lpl::core::byte{0xFFu}, lpl::core::byte{0x01u}};
    constexpr std::array<lpl::core::byte, 5> kThirtyThreeBits{lpl::core::byte{0xFFu}, lpl::core::byte{0xFFu},
                                                              lpl::core::byte{0xFFu}, lpl::core::byte{0xFFu},
                                                              lpl::core::byte{0x1Fu}};
    constexpr std::array<lpl::core::byte, 5> kThirtyTwoBits{lpl::core::byte{0xFFu}, lpl::core::byte{0xFFu},
                                                            lpl::core::byte{0xFFu}, lpl::core::byte{0xFFu},
                                                            lpl::core::byte{0x0Fu}};
    constexpr std::array<lpl::core::byte, 2> kCut{lpl::core::byte{0x80u}, lpl::core::byte{0x80u}};

    lpl::net::protocol::BitReader sixBytes{kSixBytes, 48u};
    const auto six = sixBytes.readVarint();
    test.check(!six.has_value() && six.error() == lpl::net::protocol::BitError::kVarintTooLong,
               "a varint whose fifth byte goes on is refused as too long");
    test.check(sixBytes.bitsRead() == 0u, "a refused varint leaves the cursor where it was");

    lpl::net::protocol::BitReader thirtyThree{kThirtyThreeBits, 40u};
    const auto overflowing = thirtyThree.readVarint();
    test.check(!overflowing.has_value() && overflowing.error() == lpl::net::protocol::BitError::kVarintTooLong,
               "a varint whose value needs a 33rd bit is refused, not cut to 32");

    lpl::net::protocol::BitReader thirtyTwo{kThirtyTwoBits, 40u};
    test.check(thirtyTwo.readVarint().value_or(0u) == 0xFFFFFFFFu, "the largest 32-bit varint still reads");

    lpl::net::protocol::BitReader cut{kCut, 16u};
    const auto truncated = cut.readVarint();
    test.check(!truncated.has_value() && truncated.error() == lpl::net::protocol::BitError::kUnderflow &&
                   cut.bitsRead() == 0u,
               "a varint the source cuts short is an underflow, and moves nothing");
}

LPL_TEST(a_writer_refuses_a_field_it_cannot_hold_whole_and_keeps_refusing)
{
    std::array<lpl::core::byte, 3> three{};
    lpl::net::protocol::BitWriter writer{three};

    writer.writeU16(0x1234u);
    writer.writeU16(0x5678u);
    test.check(writer.failure() == lpl::net::protocol::BitError::kOverflow,
               "a field past the end of the destination is refused as an overflow");
    test.check(writer.bitsWritten() == 16u, "a refused field writes none of its bits");
    writer.writeU8(0x9Au);
    test.check(writer.bitsWritten() == 16u && writer.failure() == lpl::net::protocol::BitError::kOverflow,
               "after a refusal, a field that would fit is refused too, and the first failure is kept");
    writer.writeBits(1u, 0u);
    test.check(writer.failure() == lpl::net::protocol::BitError::kOverflow,
               "a later refusal of another kind does not replace the first");

    std::array<lpl::core::byte, 4> four{};
    lpl::net::protocol::BitWriter varint{four};
    varint.writeVarint(0xFFFFFFFFu);
    test.check(varint.failure() == lpl::net::protocol::BitError::kOverflow && varint.bitsWritten() == 0u,
               "a five-byte varint is refused whole by four bytes");

    lpl::net::protocol::BitWriter bytes{four};
    constexpr std::array<lpl::core::byte, 5> kFive{};
    bytes.writeBytes(kFive);
    test.check(bytes.failure() == lpl::net::protocol::BitError::kOverflow && bytes.bitsWritten() == 0u,
               "five bytes are refused whole by four");

    lpl::net::protocol::BitWriter zeroWidth{four};
    zeroWidth.writeBits(1u, 0u);
    lpl::net::protocol::BitWriter wideWidth{four};
    wideWidth.writeBits(1u, 33u);
    test.check(zeroWidth.failure() == lpl::net::protocol::BitError::kInvalidWidth &&
                   wideWidth.failure() == lpl::net::protocol::BitError::kInvalidWidth,
               "a width of 0 or of 33 bits is refused on the writing side");

    lpl::net::protocol::BitWriter emptyRange{four};
    emptyRange.writeQuantizedFloat(0.5f, 1.0f, 1.0f, 8u);
    test.check(emptyRange.failure() == lpl::net::protocol::BitError::kInvalidRange && emptyRange.bitsWritten() == 0u,
               "a range whose maximum does not exceed its minimum is refused");
}

LPL_TEST(a_writer_touches_no_byte_past_its_destination)
{
    constexpr lpl::core::u8 kCanary = 0x5Au;
    std::array<lpl::core::byte, 8> memory{};
    memory.fill(lpl::core::byte{kCanary});
    lpl::net::protocol::BitWriter three{
        std::span<lpl::core::byte>{memory.data(), 3u}
    };

    three.writeU16(0xFFFFu);
    three.writeU8(0xFFu);
    const bool threeKept = memory[3] == lpl::core::byte{kCanary} && memory[7] == lpl::core::byte{kCanary};

    memory.fill(lpl::core::byte{kCanary});
    lpl::net::protocol::BitWriter five{
        std::span<lpl::core::byte>{memory.data(), 5u}
    };
    five.writeBits(0xFu, 4u);
    five.writeU32(0xFFFFFFFFu);
    const bool fiveKept = memory[5] == lpl::core::byte{kCanary} && memory[7] == lpl::core::byte{kCanary};

    test.check(!three.failure().has_value() && three.bitsWritten() == 24u, "three bytes take 24 bits exactly");
    test.check(threeKept, "a writer that fills three bytes leaves the byte after them as it was");
    test.check(!five.failure().has_value() && five.bitsWritten() == 36u, "a 32-bit field crosses into a fifth byte");
    test.check(fiveKept, "a field that crosses into a fifth byte leaves the byte after it as it was");
}

LPL_TEST(a_writer_that_resumes_keeps_the_bits_before_its_offset)
{
    std::array<lpl::core::byte, 2> wire{lpl::core::byte{0xE0u}, lpl::core::byte{kStaleByte}};
    lpl::net::protocol::BitWriter writer{wire, 3u};

    writer.writeBits(0b00001u, 5u);
    writer.writeU8(0xFFu);
    test.check(wire[0] == lpl::core::byte{0xE1u} && wire[1] == lpl::core::byte{0xFFu},
               "a writer that starts at bit 3 keeps the three bits before it");
    test.check(writer.bitsWritten() == 16u && writer.bytes().size() == 2u, "its count starts at its offset");

    lpl::net::protocol::BitWriter pastTheEnd{wire, 17u};
    test.check(pastTheEnd.failure() == lpl::net::protocol::BitError::kOverflow && pastTheEnd.bitsWritten() == 0u,
               "an offset past the destination starts the writer failed");
}

LPL_TEST(a_quantized_position_reads_back_within_one_step)
{
    constexpr float kMinimum = 0.0f;
    constexpr float kMaximum = 1000.0f;
    constexpr lpl::core::u32 kBits = 16u;
    constexpr float kStep = (kMaximum - kMinimum) / 65535.0f;
    lpl::core::u32 positions = 0u;
    lpl::core::u32 outside = 0u;
    float largestError = 0.0f;

    for (float position = kMinimum; position <= kMaximum; position += 3.1f)
    {
        std::array<lpl::core::byte, 4> wire{};
        lpl::net::protocol::BitWriter writer{wire};
        writer.writeQuantizedFloat(position, kMinimum, kMaximum, kBits);
        lpl::net::protocol::BitReader reader{writer.bytes(), writer.bitsWritten()};
        const float read = reader.readQuantizedFloat(kMinimum, kMaximum, kBits).value_or(-1.0f);
        const float error = read > position ? read - position : position - read;
        largestError = error > largestError ? error : largestError;
        outside += error <= kStep ? 0u : 1u;
        ++positions;
    }

    test.check(positions > 300u, "positions across the whole range are quantized");
    test.check(outside == 0u, "each position reads back within one step of 15 mm");
    test.measure("positions", positions);
    test.measureHexadecimal("largest_error_bits", std::bit_cast<lpl::core::u32>(largestError));
}

LPL_TEST(a_snapshot_of_entities_comes_back_entity_for_entity)
{
    std::array<lpl::core::byte, 2u + kSnapshotEntities * 32u> wire{};
    lpl::net::protocol::BitWriter writer{wire};

    writer.writeU16(static_cast<lpl::core::u16>(kSnapshotEntities));
    for (lpl::core::u32 index = 0u; index < kSnapshotEntities; ++index)
        writeSnapshotEntity(writer, makeEntity(index));

    lpl::net::protocol::BitReader reader{writer.bytes(), writer.bitsWritten()};
    const lpl::core::u32 count = reader.readU16().value_or(0u);
    lpl::core::u32 matching = 0u;
    for (lpl::core::u32 index = 0u; index < count; ++index)
    {
        SnapshotEntity entity{};
        if (readSnapshotEntity(reader, entity) && sameEntity(entity, makeEntity(index)))
            ++matching;
    }

    test.check(!writer.failure().has_value() && writer.bitsWritten() == 16u + kSnapshotEntities * 256u,
               "each entity takes its 256 bits");
    test.check(count == kSnapshotEntities && matching == kSnapshotEntities, "every entity reads back bit for bit");
    test.measure("entities", matching);
    test.measureHexadecimal("snapshot_signature", foldBytes(writer.bytes()));
}

LPL_TEST(an_input_frame_comes_back_field_for_field)
{
    std::array<lpl::core::byte, (kInputFrames * 221u + 7u) / 8u> wire{};
    lpl::net::protocol::BitWriter writer{wire};

    for (lpl::core::u32 index = 0u; index < kInputFrames; ++index)
        writeInputFrame(writer, makeFrame(index));

    lpl::net::protocol::BitReader reader{writer.bytes(), writer.bitsWritten()};
    lpl::core::u32 matching = 0u;
    for (lpl::core::u32 index = 0u; index < kInputFrames; ++index)
    {
        InputFrame frame{};
        if (readInputFrame(reader, frame) && sameFrame(frame, makeFrame(index)))
            ++matching;
    }

    test.check(!writer.failure().has_value() && writer.bitsWritten() == kInputFrames * 221u,
               "each frame takes its 221 bits, so each starts at another bit of its byte");
    test.check(matching == kInputFrames, "every frame reads back field for field");
    test.measure("frames", matching);
    test.measureHexadecimal("frames_signature", foldBytes(writer.bytes()));
}

LPL_TEST(not_a_number_takes_the_bottom_step_on_every_target)
{
    constexpr float kNotANumber = std::numeric_limits<float>::quiet_NaN();
    std::array<lpl::core::byte, 16> wire{};
    lpl::net::protocol::BitWriter writer{wire};

    writer.writeQuantizedFloat(atRunTime(kNotANumber), atRunTime(0.0f), atRunTime(1000.0f), 16u);
    writer.writeQuantizedFloat(atRunTime(kNotANumber), atRunTime(-1.0f), atRunTime(1.0f), 32u);
    writer.writeAngle(atRunTime(kNotANumber), 10u);
    lpl::net::protocol::BitReader reader{writer.bytes(), writer.bitsWritten()};
    const lpl::core::u32 sixteen = reader.readBits(16u).value_or(kNoStep);
    const lpl::core::u32 thirtyTwo = reader.readBits(32u).value_or(kNoStep);
    const lpl::core::u32 angle = reader.readBits(10u).value_or(kNoStep);

    std::array<lpl::core::byte, 4> unbounded{};
    lpl::net::protocol::BitWriter overflowing{unbounded};
    overflowing.writeQuantizedFloat(atRunTime(3.0e38f), atRunTime(-3.0e38f), atRunTime(3.0e38f), 16u);

    test.check(!writer.failure().has_value(), "a NaN is written, not refused");
    test.check(sixteen == 0u, "NaN takes the bottom step on 16 bits");
    test.check(thirtyTwo == 0u, "NaN takes the bottom step on 32 bits");
    test.check(angle == 0u, "a NaN angle takes the bottom step");
    test.check(overflowing.failure() == lpl::net::protocol::BitError::kInvalidRange && overflowing.bitsWritten() == 0u,
               "a range whose width passes the largest float is refused, so no NaN share reaches a step");
    test.measure("nan_step_16", sixteen);
    test.measure("nan_step_32", thirtyTwo);
    test.measure("nan_angle_step", angle);
}

LPL_TEST(the_top_of_a_range_takes_the_top_step_at_every_width)
{
    lpl::core::u32 wrong = 0u;
    lpl::core::u32 firstWrongWidth = 0u;

    for (lpl::core::u32 bits = 1u; bits <= 32u; ++bits)
    {
        std::array<lpl::core::byte, 4> wire{};
        lpl::net::protocol::BitWriter writer{wire};
        writer.writeQuantizedFloat(atRunTime(1000.0f), atRunTime(0.0f), atRunTime(1000.0f), bits);
        lpl::net::protocol::BitReader reader{writer.bytes(), writer.bitsWritten()};
        const auto step = reader.readBits(bits);
        if (step.has_value() && *step == lowBits(0xFFFFFFFFu, bits))
            continue;
        ++wrong;
        firstWrongWidth = firstWrongWidth == 0u ? bits : firstWrongWidth;
    }

    test.check(wrong == 0u, "the maximum of the range takes the top step on every width from 1 to 32 bits");
    test.measure("wrong_widths", wrong);
    test.measure("first_wrong_width", firstWrongWidth);
}

LPL_TEST(an_angle_too_large_to_count_its_turns_is_clamped_by_the_range)
{
    constexpr float kInfinity = std::numeric_limits<float>::infinity();
    std::array<lpl::core::byte, 8> wire{};
    lpl::net::protocol::BitWriter writer{wire};

    writer.writeAngle(atRunTime(1.0e11f), 10u);
    writer.writeAngle(atRunTime(-1.0e11f), 10u);
    writer.writeAngle(atRunTime(kInfinity), 10u);
    writer.writeAngle(atRunTime(-kInfinity), 10u);
    lpl::net::protocol::BitReader reader{writer.bytes(), writer.bitsWritten()};
    const lpl::core::u32 large = reader.readBits(10u).value_or(kNoStep);
    const lpl::core::u32 negativeLarge = reader.readBits(10u).value_or(kNoStep);
    const lpl::core::u32 infinite = reader.readBits(10u).value_or(kNoStep);
    const lpl::core::u32 negativeInfinite = reader.readBits(10u).value_or(kNoStep);

    test.check(large == 1023u && infinite == 1023u, "an angle past 2^31 turns takes the top step");
    test.check(negativeLarge == 0u && negativeInfinite == 0u, "an angle below -2^31 turns takes the bottom step");
}

LPL_TEST(a_width_read_from_the_wire_outside_one_to_thirty_two_is_refused)
{
    constexpr std::array<lpl::core::byte, 8> kSource{};
    lpl::net::protocol::BitReader reader{kSource, 64u};

    const auto zero = reader.readBits(0u);
    const auto wide = reader.readBits(33u);
    const auto quantized = reader.readQuantizedFloat(-1.0f, 1.0f, 0u);
    const auto angle = reader.readAngle(40u);

    test.check(!zero.has_value() && zero.error() == lpl::net::protocol::BitError::kInvalidWidth,
               "a width of 0 bits is refused");
    test.check(!wide.has_value() && wide.error() == lpl::net::protocol::BitError::kInvalidWidth,
               "a width of 33 bits is refused, though the source holds them");
    test.check(!quantized.has_value() && quantized.error() == lpl::net::protocol::BitError::kInvalidWidth,
               "a quantized field on 0 bits is refused, not read as 0/0");
    test.check(!angle.has_value() && angle.error() == lpl::net::protocol::BitError::kInvalidWidth,
               "an angle on 40 bits is refused");
    test.check(reader.bitsRead() == 0u, "a refused width moves nothing");
}

LPL_TEST(a_range_read_from_the_wire_that_is_not_one_is_refused)
{
    constexpr std::array<lpl::core::byte, 8> kSource{lpl::core::byte{0xABu}, lpl::core::byte{0xCDu}};
    const std::array<float, 5> extents{std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                                       0.0f, -5.0f, 3.0e38f};
    lpl::core::u32 refused = 0u;
    lpl::core::u32 moved = 0u;

    for (const float extent : extents)
    {
        lpl::net::protocol::BitReader reader{kSource, 64u};
        const float half = atRunTime(extent);
        const auto position = reader.readQuantizedFloat(-half, half, 16u);
        refused += !position.has_value() && position.error() == lpl::net::protocol::BitError::kInvalidRange ? 1u : 0u;
        moved += reader.bitsRead() != 0u ? 1u : 0u;
    }

    test.check(refused == extents.size(),
               "an extent of NaN, infinity, 0, -5 or 3e38, as a forged StateDelta sends it, is refused as no range");
    test.check(moved == 0u, "a refused range moves nothing");
    test.measure("ranges_refused", refused);
}

LPL_TEST(a_reader_holds_no_more_bits_than_its_source)
{
    std::array<lpl::core::byte, 8> memory{};
    memory.fill(lpl::core::byte{0x11u});
    const std::span<const lpl::core::byte> twoBytes{memory.data(), 2u};
    lpl::net::protocol::BitReader reader{twoBytes, 64u};

    const lpl::core::u32 remaining = reader.bitsRemaining();
    const auto pastTheSource = reader.readU32();

    test.check(remaining == 16u, "a count of 64 bits over a source of 2 bytes holds 16");
    test.check(!pastTheSource.has_value() && pastTheSource.error() == lpl::net::protocol::BitError::kUnderflow,
               "32 bits are refused rather than read past the source");
    test.check(reader.readU16().value_or(0u) == 0x1111u, "the 16 bits the source holds still read");
    test.measure("bits_held", remaining);
}
