# KittenFish NNUE training

Original C++17 CPU NNUE baseline: training, quantization, incremental evaluation,
model IO and benchmarks. **No search, UCI, move generator or trained weights.**
This is Stockfish-inspired, not a copy of Stockfish's network or file format.

## Architecture

- 16 king buckets (2x2 square regions) per perspective.
- 12 relative piece types x 64 oriented squares per bucket; kings included.
- Shared 12,288 x 864 sparse feature transformer and 864 biases.
- Two 864-element accumulators: White and Black perspectives.
- Clipped ReLU [0,1], concatenated side-to-move first, linear scalar output.
- 10,619,425 trainable parameters. Float training; int16 weights and int32
  accumulators for deployment, with int64 output sum. No SIMD intrinsics needed;
  the optimized compiler can vectorize loops.
- King bucket changes rebuild the affected perspective. Other changes subtract
  removed pieces and add new pieces. Final-board updates cover captures,
  castling, en passant and promotions without special move encodings.

The small output head is intentionally simpler than modern Stockfish. Strength
and 100k single-thread evaluations/s are targets, not measured guarantees.
The update API scans 64 squares; a future engine can pass explicit dirty pieces.

## Build and test

Linux/macOS, GNU Make and a C++17 compiler; no third-party libraries:

```sh
make
make test
```

`-march=native` targets the build CPU. For a portable binary:
`make CXXFLAGS='-O3 -std=c++17 -Wall -Wextra -Wpedantic'`.

## Training data contract

Uncompressed TSV, no header: full FEN, one tab, numeric centipawns **from White's
perspective**. Blank lines and lines beginning with # are ignored.

```text
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1	20
```

Use an actual tab, not the two characters backslash+t. Mate scores must be
converted upstream using an explicit chosen convention, or excluded. Labels
are clipped to +/-2000cp, converted to side-to-move, and divided by 400.
Lichess JSONL conversion is not included; do not pass its raw archive here.
Prepare independent training/validation files, deduplicate positions across
splits, and group by source game when source identity is available.

```sh
./build/kfnnue train train.tsv val.tsv kittenfish 10 0.0001
# Resume float weights (optimizer is plain SGD, no momentum state):
./build/kfnnue train train.tsv val.tsv continued 10 0.0001 kittenfish.last.kff
./build/kfnnue export kittenfish.best.kff kittenfish.nnue
./build/kfnnue eval kittenfish.nnue 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
./build/kfnnue bench kittenfish.nnue 1000000
```

Online sparse SGD, deterministic epoch shuffling (seed 42), MSE loss,
clipped output gradient, bounded parameters. Validation prints float-model
MSE in cp squared and MAE in cp. The best float validation MSE selects
`.best.kff` and quantized `.nnue`; `.last.kff` supports continuation. On resume,
shuffle order and best-metric tracking restart. Quantized validation should be
measured separately before deployment; quantization can change accuracy.

This is a correctness-first CPU trainer, not a high-throughput GPU training
pipeline. It loads datasets into RAM and updates one sample at a time. Do not
load hundreds of millions of rows at once. Use curated subsets initially.

## Benchmarks and API

`bench` measures a single thread, printing separate cached-accumulator forward
throughput and incremental-update-plus-eval throughput. It alternates start
and e4 positions, so it is a warm-cache microbenchmark, not a search NPS claim
or a diverse-position benchmark. Full FEN parsing/rebuild cost is excluded.
Benchmark your eventual target CPU; no speed promise is implied.

`QuantNet::start(position)` builds state; `update(state, next_position)` updates
it; `eval(state)` returns centipawns from side-to-move perspective. Copy a State
before an update to implement restore/undo later. One network is read-only and
can be shared; each worker must own its State. This code validates basic FEN
syntax, both kings and piece count, but does not validate legal chess positions.
The evaluator learns static position scores; it does not adjudicate draws,
checkmate, repetition, castling rights or the fifty-move rule.

Model formats KFQ1 and KFF1 are fixed-architecture native-endian files, not
Stockfish-compatible .nnue files. Use trusted local models on the same endian
architecture. There is no checksum, compression, or architecture conversion.

`selftest` checks 200 accumulator updates against complete rebuilds (including
king bucket changes), one SGD descent step, and model roundtripping. Its board
mutations intentionally exercise tensor bookkeeping rather than legal play.
