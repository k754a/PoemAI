# ---------------------------------------------------------
# UPDATED SERVER SIDE FOR NEW MODEL FORMAT
#
# Matches the newer local loader:
#   - poem_model.txt
#   - poem_weights.bin
#   - context-based input weights
#   - repeat penalty instead of used-word banning
#
# Designed to run better on a small VPS by using memmap
# instead of forcing the whole binary into RAM.
# ---------------------------------------------------------

import os

# ---------------------------------------------------------
# KEEP NUMPY FROM SPAWNING TOO MANY THREADS
# ---------------------------------------------------------
# Set these BEFORE importing numpy.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np


# ---------------------------------------------------------
# SMALL ENV HELPERS
# ---------------------------------------------------------

def _env_bool(name, default=True):
    value = os.environ.get(name)

    if value is None:
        return default

    value = value.strip().lower()

    return value not in (
        "0",
        "false",
        "no",
        "off",
    )


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.environ.get(
    "POEM_MODEL_PATH",
    os.path.join(
        BASE_DIR,
        "poem_model.txt"
    )
)

WEIGHTS_PATH = os.environ.get(
    "POEM_WEIGHTS_PATH",
    os.path.join(
        BASE_DIR,
        "poem_weights.bin"
    )
)


# ---------------------------------------------------------
# SERVER / GENERATION SETTINGS
# ---------------------------------------------------------

# Use memory mapping by default to keep RSS lower.
USE_MEMMAP = _env_bool(
    "POEM_USE_MEMMAP",
    True
)

# Your local code adds tiny noise. Keep this True if you want
# the same behavior as your local script.
USE_NOISE = _env_bool(
    "POEM_USE_NOISE",
    True
)

# If True, uses the exact local-style top-K argsort path.
# If False, uses argpartition, which is faster.
EXACT_TOPK = _env_bool(
    "POEM_EXACT_TOPK",
    False
)

# ---------------------------------------------------------
# DIVERSITY SETTINGS
# ---------------------------------------------------------

# If True, choose randomly from several top candidates.
# This prevents the same poem every time.
USE_TOP_CHOICE = _env_bool(
    "POEM_USE_TOP_CHOICE",
    True
)

# How many top candidates to choose from.
try:
    TOP_CHOICE_COUNT = int(
        os.environ.get(
            "POEM_TOP_CHOICE_COUNT",
            "5"
        )
    )
except ValueError:
    TOP_CHOICE_COUNT = 5

TOP_CHOICE_COUNT = max(
    1,
    TOP_CHOICE_COUNT
)

# Local used 0.001.
# If it still feels too repetitive, try 0.005 or 0.01.
try:
    NOISE_SCALE = float(
        os.environ.get(
            "POEM_NOISE_SCALE",
            "0.001"
        )
    )
except ValueError:
    NOISE_SCALE = 0.001

# Set POEM_SEED="" in environment if you want random poems.
# Leave it as an integer if you want repeatable poems.
_SEED_ENV = os.environ.get(
    "POEM_SEED",
    ""
).strip()

try:
    SEED = int(_SEED_ENV) if _SEED_ENV else None
except ValueError:
    SEED = None

# RandomState wants a 32-bit unsigned seed.
if SEED is not None:
    SEED = SEED % (2 ** 32)

# If output weights are small enough, load them into RAM for
# faster generation. Input/context weights stay memmapped.
_LIMIT_ENV = os.environ.get(
    "POEM_OUTPUT_RAM_LIMIT_BYTES",
    ""
).strip()

try:
    OUTPUT_RAM_LIMIT_BYTES = (
        int(_LIMIT_ENV)
        if _LIMIT_ENV
        else 256 * 1024 * 1024
    )
except ValueError:
    OUTPUT_RAM_LIMIT_BYTES = 256 * 1024 * 1024


# ---------------------------------------------------------
# MATCH YOUR LOCAL SETTINGS
# ---------------------------------------------------------

MAX_TOKENS = 140
TOP_K = 30

RHYME_BOOST = np.float32(0.25)
REPEAT_PENALTY = np.float32(0.75)


# ---------------------------------------------------------
# LOAD MODEL FILE
# ---------------------------------------------------------

print("Loading model...")

if not os.path.exists(MODEL_PATH):
    print(
        f"ERROR: {MODEL_PATH} not found!"
    )
    raise SystemExit(1)

if not os.path.exists(WEIGHTS_PATH):
    print(
        f"ERROR: {WEIGHTS_PATH} not found!"
    )
    raise SystemExit(1)


with open(
    MODEL_PATH,
    encoding="utf-8",
    errors="ignore"
) as f:

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    word_count = int(f.readline())
    neuron_count = int(f.readline())
    context = int(f.readline())
    vocab_size = int(f.readline())

    if word_count <= 0:
        print("ERROR: word_count must be > 0")
        raise SystemExit(1)

    if neuron_count <= 0:
        print("ERROR: neuron_count must be > 0")
        raise SystemExit(1)

    if context <= 0:
        print("ERROR: context must be > 0")
        raise SystemExit(1)

    if vocab_size < 0:
        print("ERROR: vocab_size must be >= 0")
        raise SystemExit(1)

    print(f"Word count:   {word_count}")
    print(f"Neuron count: {neuron_count}")
    print(f"Context:      {context}")
    print(f"Vocab size:   {vocab_size}")

    # -----------------------------------------------------
    # RHYME GROUPS
    # -----------------------------------------------------

    line = f.readline().strip()

    if line != "RHYME_GROUPS":
        print(
            f"ERROR: Expected RHYME_GROUPS, got {line!r}"
        )
        raise SystemExit(1)

    rhyme_compatible = {}

    while True:

        line = f.readline().strip()

        if line == "END_RHYME":
            break

        if not line:
            continue

        parts = line.split()

        if len(parts) < 2:
            continue

        try:
            ids = [
                int(x)
                for x in parts[1:]
            ]
        except ValueError:
            continue

        if len(ids) < 2:
            continue

        for wid in ids:

            if wid not in rhyme_compatible:
                rhyme_compatible[wid] = set()

            rhyme_compatible[wid].update(ids)

    # -----------------------------------------------------
    # VOCABULARY
    # -----------------------------------------------------

    id2word = {}
    word2id = {}

    for _ in range(vocab_size):

        line = f.readline().strip()

        if not line:
            continue

        parts = line.split(
            " ",
            1
        )

        if len(parts) < 2:
            continue

        # Smart parsing:
        #   ID WORD
        # or
        #   WORD ID
        try:
            wid = int(parts[0])
            word = parts[1]
        except ValueError:
            try:
                wid = int(parts[1])
                word = parts[0]
            except ValueError:
                continue

        id2word[wid] = word
        word2id[word] = wid


# ---------------------------------------------------------
# CLEAN RHYME GROUPS
# ---------------------------------------------------------
# Convert sets to tuples to reduce overhead and make them
# read-only. Also filter invalid IDs now.

clean_rhymes = {}

for wid, ids in rhyme_compatible.items():

    if 0 < wid < word_count:

        clean_ids = tuple(
            x
            for x in ids
            if 0 < x < word_count
        )

        if clean_ids:
            clean_rhymes[wid] = clean_ids

rhyme_compatible = clean_rhymes


# ---------------------------------------------------------
# WEIGHT FILE SIZE CHECK
# ---------------------------------------------------------

input_values = (
    context
    * word_count
    * neuron_count
)

output_values = (
    word_count
    * neuron_count
)

total_values = (
    input_values
    + output_values
)

expected_bytes = total_values * 4
actual_bytes = os.path.getsize(WEIGHTS_PATH)

print(
    f"Expected weight file: "
    f"{expected_bytes / (1024 ** 3):.2f} GB"
)

print(
    f"Actual weight file:   "
    f"{actual_bytes / (1024 ** 3):.2f} GB"
)

if actual_bytes != expected_bytes:

    print(
        "ERROR: Binary weight file size does not match "
        "the model dimensions!"
    )

    print(f"Expected: {expected_bytes:,} bytes")
    print(f"Actual:   {actual_bytes:,} bytes")

    raise SystemExit(1)


# ---------------------------------------------------------
# LOAD BINARY WEIGHTS
# ---------------------------------------------------------

print("Loading binary weights...")

OUTPUT_IN_RAM = False

if USE_MEMMAP:

    # -----------------------------------------------------
    # INPUT WEIGHTS: MEMMAP
    # -----------------------------------------------------
    # Shape:
    #   (context, word_count, neuron_count)
    #
    # This matches:
    #   weights[position, word_id]
    #
    # -----------------------------------------------------

    weights = np.memmap(
        WEIGHTS_PATH,
        dtype=np.float32,
        mode="r",
        offset=0,
        shape=(
            context,
            word_count,
            neuron_count
        )
    )

    # -----------------------------------------------------
    # OUTPUT WEIGHTS
    # -----------------------------------------------------
    # If they are small enough, load them fully into RAM.
    # Otherwise memmap them too.
    #
    # Output shape:
    #   (word_count, neuron_count)
    # -----------------------------------------------------

    output_bytes = output_values * 4
    output_offset = input_values * 4

    if output_bytes <= OUTPUT_RAM_LIMIT_BYTES:

        try:

            out_flat = np.fromfile(
                WEIGHTS_PATH,
                dtype=np.float32,
                count=output_values,
                offset=output_offset
            )

            out_weights = out_flat.reshape(
                word_count,
                neuron_count
            )

            del out_flat

            OUTPUT_IN_RAM = True

        except Exception as e:

            print(
                "WARNING: Could not load output weights "
                "into RAM. Falling back to memmap."
            )

            print(str(e))

            out_weights = np.memmap(
                WEIGHTS_PATH,
                dtype=np.float32,
                mode="r",
                offset=output_offset,
                shape=(
                    word_count,
                    neuron_count
                )
            )

    else:

        out_weights = np.memmap(
            WEIGHTS_PATH,
            dtype=np.float32,
            mode="r",
            offset=output_offset,
            shape=(
                word_count,
                neuron_count
            )
        )

else:

    # -----------------------------------------------------
    # LOAD EVERYTHING INTO RAM
    # -----------------------------------------------------
    # This is faster but can use a lot of memory.
    # -----------------------------------------------------

    all_weights = np.fromfile(
        WEIGHTS_PATH,
        dtype=np.float32
    )

    if all_weights.size != total_values:

        print(
            "ERROR: Loaded the wrong number of weight values!"
        )

        print(f"Expected: {total_values:,}")
        print(f"Loaded:   {all_weights.size:,}")

        raise SystemExit(1)

    weights = all_weights[:input_values].reshape(
        context,
        word_count,
        neuron_count
    )

    out_weights = all_weights[input_values:].reshape(
        word_count,
        neuron_count
    )

    del all_weights

    OUTPUT_IN_RAM = True


# ---------------------------------------------------------
# GLOBAL MODEL SETTINGS
# ---------------------------------------------------------

CONTEXT = context


# ---------------------------------------------------------
# COMMON WORD FALLBACK
# ---------------------------------------------------------

common_ids = np.arange(
    1,
    min(
        500,
        word_count
    ),
    dtype=np.int64
)

if len(common_ids) == 0:

    common_ids = np.arange(
        1,
        min(
            10,
            word_count
        ),
        dtype=np.int64
    )


# ---------------------------------------------------------
# PRECOMPUTED WORD BIAS
# ---------------------------------------------------------
# Local code did this every token:
#
#   scores[1:] -= np.arange(1, word_count) * 0.000001
#
# We build it once.

word_bias = np.zeros(
    word_count,
    dtype=np.float32
)

if word_count > 1:

    word_bias[1:] = (
        np.arange(
            1,
            word_count,
            dtype=np.float32
        )
        * np.float32(0.000001)
    )


# ---------------------------------------------------------
# PRINT LOAD SUMMARY
# ---------------------------------------------------------

print()

print(
    f"Model loaded: {len(id2word)} vocab entries, "
    f"{word_count} word rows, "
    f"{neuron_count} neurons, "
    f"context {CONTEXT}"
)

print(
    f"Rhyme words loaded: {len(rhyme_compatible)}"
)

print(
    f"Input weights shape:  {weights.shape}"
)

print(
    f"Output weights shape: {out_weights.shape}"
)

print(
    f"Input weights mode:  "
    f"{'memmap' if USE_MEMMAP else 'RAM'}"
)

print(
    f"Output weights mode: "
    f"{'RAM' if OUTPUT_IN_RAM else 'memmap'}"
)

print()


# ---------------------------------------------------------
# GENERATE POEM
# ---------------------------------------------------------

def generate_poem(text):

    # -----------------------------------------------------
    # SAFETY
    # -----------------------------------------------------

    if text is None:
        return ""

    if word_count <= 1:
        return ""

    if neuron_count <= 0:
        return ""

    # -----------------------------------------------------
    # CLEAN INPUT
    # -----------------------------------------------------

    text = text.strip().lower()

    if not text:
        return ""

    prompt_words = text.split()

    if not prompt_words:
        return ""

    # -----------------------------------------------------
    # REQUEST-LOCAL RNG
    # -----------------------------------------------------
    # This keeps Flask/thread requests isolated.
    #
    # If SEED is an integer, the same prompt gives the same
    # poem every time.
    #
    # If SEED is None, poems vary.
    # -----------------------------------------------------

    if SEED is None:
        rng = np.random.RandomState()
    else:
        rng = np.random.RandomState(SEED)

    # -----------------------------------------------------
    # REQUEST-LOCAL BUFFERS
    # -----------------------------------------------------

    hidden = np.zeros(
        neuron_count,
        dtype=np.float32
    )

    scores = np.empty(
        word_count,
        dtype=np.float32
    )

    # -----------------------------------------------------
    # FIRST WORD
    # -----------------------------------------------------

    first_word = prompt_words[0]

    first_id = word2id.get(
        first_word,
        0
    )

    # Treat out-of-range IDs as unknown too.
    if (
        first_id <= 0
        or first_id >= word_count
    ):

        if len(common_ids) == 0:
            return ""

        first_id = int(
            rng.choice(common_ids)
        )

        first_word = id2word.get(
            first_id,
            "the"
        )

    # -----------------------------------------------------
    # INITIAL STATE
    # -----------------------------------------------------

    generated_words = [
        first_word
    ]

    ids = [
        first_id
    ]

    word_usage = {
        first_id: 1
    }

    rhyme_target_id = None

    # -----------------------------------------------------
    # GENERATION LOOP
    # -----------------------------------------------------

    for step in range(MAX_TOKENS - 1):

        # -------------------------------------------------
        # RECENT CONTEXT
        # -------------------------------------------------

        current_ids = ids[-CONTEXT:]

        # -------------------------------------------------
        # BUILD HIDDEN STATE
        # -------------------------------------------------
        # This intentionally uses a Python loop to match
        # your local code exactly:
        #
        #   hidden += weights[position, wid]
        #
        # That preserves position-specific context behavior.
        # -------------------------------------------------

        hidden.fill(0.0)

        for position, wid in enumerate(current_ids):

            if position >= CONTEXT:
                break

            if (
                wid > 0
                and wid < word_count
            ):
                hidden += weights[position, wid]

        np.clip(
            hidden,
            -1.0,
            1.0,
            out=hidden
        )

        # -------------------------------------------------
        # SCORE WORDS
        # -------------------------------------------------

        np.dot(
            out_weights,
            hidden,
            out=scores
        )

        # Clean NaN/Inf like local code.
        scores = np.nan_to_num(
            scores,
            copy=False,
            nan=0.0,
            posinf=1.0e10,
            neginf=-1.0e10
        )

        scores = scores.astype(
            np.float32,
            copy=False
        )

        # -------------------------------------------------
        # BAN PADDING TOKEN
        # -------------------------------------------------

        scores[0] = -np.inf

        # -------------------------------------------------
        # WORD BIAS
        # -------------------------------------------------

        scores -= word_bias

        # -------------------------------------------------
        # REPETITION PENALTY
        # -------------------------------------------------
        # Your local code penalizes used words.
        # It does NOT ban them.
        # -------------------------------------------------

        for used_id, count in word_usage.items():

            if (
                used_id > 0
                and used_id < word_count
            ):

                scores[used_id] -= (
                    REPEAT_PENALTY
                    * count
                )

        # -------------------------------------------------
        # RHYME BOOST
        # -------------------------------------------------

        if rhyme_target_id is not None:

            compatible_ids = rhyme_compatible.get(
                rhyme_target_id,
                ()
            )

            for wid in compatible_ids:

                if (
                    wid > 0
                    and wid < word_count
                ):
                    scores[wid] += RHYME_BOOST

        # -------------------------------------------------
        # TINY RANDOMNESS
        # -------------------------------------------------

        if USE_NOISE:

            noise = rng.normal(
                0.0,
                NOISE_SCALE,
                size=scores.shape
            )

            scores += noise.astype(np.float32)

        # -------------------------------------------------
        # GET TOP CANDIDATES
        # -------------------------------------------------

        candidate_count = min(
            TOP_K,
            word_count - 1
        )

        if candidate_count <= 0:
            break

        if EXACT_TOPK:

            top_indices = np.argsort(
                scores
            )[-candidate_count:][::-1]

        else:

            top_indices = np.argpartition(
                scores,
                -candidate_count
            )[-candidate_count:]

            top_indices = top_indices[
                np.argsort(
                    scores[top_indices]
                )[::-1]
            ]

        valid = []

        for idx in top_indices:

            idx = int(idx)

            if idx <= 0:
                continue

            if idx >= word_count:
                continue

            if not np.isfinite(scores[idx]):
                continue

            valid.append(idx)

        if not valid:
            break

        # -------------------------------------------------
        # PICK WORD
        # -------------------------------------------------

        if USE_TOP_CHOICE:

            choice_count = min(
                TOP_CHOICE_COUNT,
                len(valid)
            )

            best_id = int(
                rng.choice(
                    valid[:choice_count]
                )
            )

        else:

            best_id = valid[0]

        # -------------------------------------------------
        # GET WORD
        # -------------------------------------------------

        next_word = id2word.get(
            best_id,
            "?"
        )

        if next_word == "?":
            break

        # -------------------------------------------------
        # NEWLINE HANDLING
        # -------------------------------------------------

        if next_word == "<NEWLINE>":

            # Do not allow two newlines in a row.
            if (
                generated_words
                and generated_words[-1] == "<NEWLINE>"
            ):
                continue

            # Find the last real word of this line.
            previous_word_id = None

            for word in reversed(generated_words):

                if word == "<NEWLINE>":
                    break

                wid = word2id.get(word)

                if wid is not None:
                    previous_word_id = wid
                    break

            # That word becomes the rhyme target for the
            # next line.
            if previous_word_id is not None:
                rhyme_target_id = previous_word_id

            generated_words.append("<NEWLINE>")
            ids.append(best_id)

            # IMPORTANT:
            # Do NOT add <NEWLINE> to word_usage.
            # Your local code does not count it.
            continue

        # -------------------------------------------------
        # NORMAL WORD
        # -------------------------------------------------

        generated_words.append(next_word)
        ids.append(best_id)

        word_usage[best_id] = (
            word_usage.get(best_id, 0) + 1
        )

    # -----------------------------------------------------
    # FORMAT POEM
    # -----------------------------------------------------

    lines = []
    current_line = []

    for word in generated_words:

        if word == "<NEWLINE>":

            if current_line:
                lines.append(
                    " ".join(current_line)
                )

            current_line = []

        else:

            current_line.append(word)

    if current_line:
        lines.append(
            " ".join(current_line)
        )

    return "\n".join(lines)


# ---------------------------------------------------------
# OPTIONAL LOCAL TEST MODE
# ---------------------------------------------------------
# If you run this file directly, you can test generation.
# If you import it from Flask, this does nothing.

if __name__ == "__main__":

    print("--- AI Poet Server Test ---")
    print("Type 'exit' to stop.")
    print()

    while True:

        try:
            text = input("> ").strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):
            print("\nbye.")
            break

        if text.lower() == "exit":
            break

        poem = generate_poem(text)

        print()
        print("AI Poem:")
        print(poem if poem else "")
        print()