#!/usr/bin/env python3
"""Usage: tools/erasure-model.py [--frames N]

Models how a stream of one message per tick, at 60 Hz, survives the loss of datagrams, for each way of surviving it
that LplPlugin#239 weighs, and prints one CSV row per loss model, deadline and scheme. A model, not the channel: it
decides for every message when the receiver has it, and what the scheme puts on the wire. It needs no payload: a lost
message is an unknown, a parity row is a vector of GF(256) coefficients over the unknowns of its window, and a message
is recovered once the rows the receiver holds span it. Every run with the same N prints the same table.

The loss models are Gilbert channels, every datagram sent in the bad state lost and none in the good one: 10 % of
datagrams lost independently, 10 % in bursts of 2 and of 4 on average, and 1 % independently. A deadline is the tick
by which a message must be in hand, counted from its own: 2, 4 and 8 ticks, or 33, 67 and 133 ms.

The schemes put one datagram on the wire per tick, except a resend:
  none                    the message alone.
  repeat the last C       the message and the C - 1 before it.
  parity R over W         R rows, each a random non-zero coefficient of GF(256) per message of the W before, as the
                          RLC codes of RFC 8681; the sender computes every row from scratch.
  binary parity R over W  each row the XOR of each message of the window with probability one half (GF(2)).
  power parity R over W   each coefficient one of 1, a, a^2 and a^3, which a sender applies by doublings.
  half-dense, quarter-dense power parity
                          as power parity, each message kept in a row with probability 1/2 or 1/4.
  Vandermonde parity R over W
                          row j gives the message of tick m the coefficient a^(j*m), the same in every datagram, so a
                          sender keeps each row by a sliding update; consecutive datagrams' rows are then multiples
                          of one another, except at the window's edges.
  interleaved XOR R over W
                          row j the XOR of the messages of the window whose tick is j modulo R.
  resend, round trip T    a lost message is asked for once a later one arrives and lands a round trip later, asked
                          again until it gets through: two more datagrams an attempt.

Columns: the loss model, the scheme, the deadline in ticks and in ms; the messages not in hand by their deadline, and
per million; the floor no scheme of one datagram a tick can go under, per million: a message is lost for good when
it and every datagram until its deadline are lost, p * s^D, s the chance a loss follows a loss; the 99th percentile
and the largest delay of a recovered message, in ticks; the bytes a tick puts on the wire for messages of 16, 64 and
200 bytes, Ethernet, IPv4 and UDP headers and a 4-byte sequence number included, each row 2 bytes more than a
message; the datagrams a tick.

Exit status: 0 when the table is printed, 2 on a usage error.
"""

import argparse
import random
import sys

TICK_MS = 1000.0 / 60.0
HEADERS = 14 + 20 + 8
CHANNEL_HEADER = 4
LOSSES = (("iid 10 %", 0.10, 1.0 / 0.9), ("bursts of 2 at 10 %", 0.10, 2.0), ("bursts of 4 at 10 %", 0.10, 4.0),
          ("iid 1 %", 0.01, 1.0 / 0.99))
DEADLINES = (2, 4, 8)
SIZES = (16, 64, 200)
ROUND_TRIPS_MS = (50, 100, 200)
SCHEMES = (("none", 0, 0), ("repeat", 2, 0), ("repeat", 3, 0), ("repeat", 4, 0),
           ("random", 4, 1), ("random", 8, 1), ("random", 4, 2), ("random", 8, 2), ("random", 16, 2),
           ("random", 16, 3), ("binary", 8, 2), ("binary", 8, 3), ("powers", 8, 2), ("sparse2", 8, 2),
           ("sparse2", 8, 3), ("sparse4", 8, 3), ("vandermonde", 8, 2), ("vandermonde", 8, 3),
           ("interleaved", 8, 2), ("interleaved", 16, 3), ("interleaved", 16, 4))
LABELS = {"random": "parity", "binary": "binary parity", "powers": "power parity",
          "sparse2": "half-dense power parity", "sparse4": "quarter-dense power parity",
          "vandermonde": "Vandermonde parity", "interleaved": "interleaved XOR"}


def build_field():
    """GF(256) over x^8 + x^4 + x^3 + x^2 + 1 (0x11D), a = 2: its exponentials, doubled in length, and logarithms."""
    exponentials = [0] * 512
    logarithms = [0] * 256
    value = 1
    for power in range(255):
        exponentials[power] = value
        logarithms[value] = power
        value <<= 1
        if value & 0x100:
            value ^= 0x11D
    for power in range(255, 512):
        exponentials[power] = exponentials[power - 255]
    return exponentials, logarithms


EXP, LOG = build_field()


def multiply(a, b):
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]]


def inverse(a):
    return EXP[255 - LOG[a]]


class GilbertChannel:
    """Every datagram sent in the bad state is lost, none in the good one; bursts last `burst` datagrams on average."""

    def __init__(self, rate, burst, draws):
        self.leave_bad = 1.0 / burst
        self.enter_bad = rate * self.leave_bad / (1.0 - rate)
        self.draws = draws
        self.bad = draws.random() < rate

    def loses(self):
        lost = self.bad
        if self.bad:
            self.bad = self.draws.random() >= self.leave_bad
        else:
            self.bad = self.draws.random() < self.enter_bad
        return lost


def eliminate(target, pivot, source):
    factor = target.get(pivot, 0)
    if not factor:
        return
    for column, value in source.items():
        merged = target.get(column, 0) ^ multiply(factor, value)
        if merged:
            target[column] = merged
        else:
            target.pop(column, None)


class Rows:
    """Rows over the unknown messages, kept fully reduced, so that a solved unknown is a row of one entry."""

    def __init__(self):
        self.by_pivot = {}

    def add(self, row):
        row = dict(row)
        for pivot, held in self.by_pivot.items():
            eliminate(row, pivot, held)
        if not row:
            return
        pivot = min(row)
        scale = inverse(row[pivot])
        row = {column: multiply(scale, value) for column, value in row.items()}
        for held in self.by_pivot.values():
            eliminate(held, pivot, row)
        self.by_pivot[pivot] = row

    def solves(self, unknown):
        row = self.by_pivot.get(unknown)
        return row is not None and len(row) == 1

    def learn(self, known):
        self.by_pivot.pop(known, None)
        for row in self.by_pivot.values():
            row.pop(known, None)


def rows_of_datagram(kind, draws, tick, window, parity, unknown):
    """The rows the datagram of `tick` carries, restricted to the messages the receiver lacks."""
    span = [m for m in range(tick - window, tick) if m in unknown]
    if kind == "random":
        return [{m: draws.randrange(1, 256) for m in span} for _ in range(parity)]
    if kind == "binary":
        return [{m: 1 for m in span if draws.random() < 0.5} for _ in range(parity)]
    if kind == "powers":
        return [{m: EXP[draws.randrange(0, 4)] for m in span} for _ in range(parity)]
    if kind in ("sparse2", "sparse4"):
        keep = 1.0 / int(kind[-1])
        return [{m: EXP[draws.randrange(0, 4)] for m in span if draws.random() < keep} for _ in range(parity)]
    if kind == "vandermonde":
        return [{m: EXP[(j * m) % 255] for m in span} for j in range(parity)]
    return [{m: 1 for m in span if m % parity == j} for j in range(parity)]


def run_rows(frames, channel, draws, kind, window, parity, deadline):
    """A message past its deadline stops being wanted but stays an unknown: the rows that name it still need it."""
    rows = Rows()
    wanted = set()
    expired = set()
    delays = []
    missed = 0
    for tick in range(frames):
        if channel.loses():
            wanted.add(tick)
        else:
            delays.append(0)
            if window and wanted:
                for row in rows_of_datagram(kind, draws, tick, window, parity, wanted | expired):
                    if row:
                        rows.add(row)
                for m in sorted(wanted):
                    if rows.solves(m):
                        delays.append(tick - m)
                        wanted.discard(m)
                        rows.learn(m)
        for m in [m for m in wanted if tick - m >= deadline]:
            missed += 1
            wanted.discard(m)
            expired.add(m)
        for m in [m for m in expired if m < tick - window]:
            expired.discard(m)
            rows.by_pivot.pop(m, None)
    return missed + len(wanted), delays


def run_repeat(frames, channel, copies, deadline):
    delays = []
    missed = 0
    pending = []
    for tick in range(frames):
        if channel.loses():
            pending.append(tick)
            continue
        delays.append(0)
        for m in pending:
            if tick - m < copies and tick - m <= deadline:
                delays.append(tick - m)
            else:
                missed += 1
        pending = []
    return missed + len(pending), delays


def run_resend(frames, forward, backward, round_trip_ticks, deadline):
    delays = []
    missed = 0
    extra = 0
    pending = []
    for tick in range(frames):
        if forward.loses():
            pending.append(tick)
            continue
        delays.append(0)
        for m in pending:
            at = tick
            while True:
                extra += 2
                at += round_trip_ticks
                if not backward.loses() and not forward.loses():
                    break
            if at - m <= deadline:
                delays.append(at - m)
            else:
                missed += 1
        pending = []
    return missed + len(pending), delays, extra


def percentile(values, share):
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))]


def scheme_name(kind, size, parity):
    if kind == "none":
        return "none"
    if kind == "repeat":
        return f"repeat the last {size}"
    return f"{LABELS[kind]} {parity} over {size}"


def wire_bytes(kind, size, parity, message):
    if kind == "none":
        return HEADERS + CHANNEL_HEADER + message
    if kind == "repeat":
        return HEADERS + CHANNEL_HEADER + size * message + (size - 1)
    return HEADERS + CHANNEL_HEADER + message + parity * (message + 2)


def write_row(out, loss, name, deadline, frames, missed, floor, delays, bytes_per_tick, datagrams):
    recovered = [d for d in delays if d > 0]
    out.write(f"{loss},{name},{deadline},{deadline * TICK_MS:.0f},{missed},{missed * 1000000 // frames},"
              f"{floor:.3f},{percentile(recovered, 0.99)},{max(recovered or [0])},"
              + ",".join(f"{b:.1f}" for b in bytes_per_tick) + f",{datagrams:.3f}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("--frames", type=int, default=1000000, help="messages per row of the table (1,000,000)")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames takes a positive count")
    out = sys.stdout
    out.write("loss,scheme,deadline_ticks,deadline_ms,missed,missed_per_million,floor_per_million,"
              "p99_recovery_ticks,max_recovery_ticks," + ",".join(f"bytes_per_tick_{s}" for s in SIZES)
              + ",datagrams_per_tick\n")
    for index, (loss, rate, burst) in enumerate(LOSSES):
        seed = 0x2390 + index * 16
        stay_lost = 1.0 - 1.0 / burst
        for deadline in DEADLINES:
            floor = rate * stay_lost ** deadline * 1000000
            for kind, size, parity in SCHEMES:
                channel = GilbertChannel(rate, burst, random.Random(seed))
                if kind == "repeat":
                    missed, delays = run_repeat(args.frames, channel, size, deadline)
                else:
                    missed, delays = run_rows(args.frames, channel, random.Random(seed ^ 0xC0EF), kind, size, parity,
                                              deadline)
                write_row(out, loss, scheme_name(kind, size, parity), deadline, args.frames, missed, floor, delays,
                          [wire_bytes(kind, size, parity, s) for s in SIZES], 1.0)
            for round_trip_ms in ROUND_TRIPS_MS:
                forward = GilbertChannel(rate, burst, random.Random(seed))
                backward = GilbertChannel(rate, burst, random.Random(seed ^ 0xBAC4))
                missed, delays, extra = run_resend(args.frames, forward, backward,
                                                   max(1, round(round_trip_ms / TICK_MS)), deadline)
                share = extra / args.frames
                bytes_per_tick = [HEADERS + CHANNEL_HEADER + s + share / 2 * (HEADERS + CHANNEL_HEADER)
                                  + share / 2 * (HEADERS + CHANNEL_HEADER + s) for s in SIZES]
                write_row(out, loss, f"resend round trip {round_trip_ms} ms", deadline, args.frames, missed, floor,
                          delays, bytes_per_tick, 1.0 + share)
            out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
