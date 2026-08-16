# SPED UP USING GPT, TO RUN WELL ON NEST, WIHOUT USING 2 GB OF RAM.


import os

# ---------------------------------------------------------
# KEEP NUMPY FROM GOING CRAZY WITH THREADS
# ---------------------------------------------------------
# Set these BEFORE importing numpy.
# Your server only has 2 cores, so don't let BLAS summon 47
# billion threads and turn the VPS into a microwave.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "poem_model.txt"
)

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "poem_weights.bin"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("Loading model...")


if not os.path.exists(
    MODEL_PATH
):

    print(
        "ERROR: poem_model.txt not found. "
        "Train the C++ model first!"
    )

    raise SystemExit(1)


if not os.path.exists(
    WEIGHTS_PATH
):

    print(
        "ERROR: poem_weights.bin not found. "
        "Train the C++ model first!"
    )

    raise SystemExit(1)


# ---------------------------------------------------------
# LOAD MODEL METADATA
# ---------------------------------------------------------

with open(
    MODEL_PATH,
    encoding="utf-8",
    errors="ignore"
) as f:

    word_count = int(
        f.readline()
    )

    neuron_count = int(
        f.readline()
    )

    context = int(
        f.readline()
    )

    vocab_size = int(
        f.readline()
    )


    print(
        f"Word count:   {word_count}"
    )

    print(
        f"Neuron count: {neuron_count}"
    )

    print(
        f"Context:      {context}"
    )

    print(
        f"Vocab size:   {vocab_size}"
    )


    # -----------------------------------------------------
    # RHYME GROUPS
    # -----------------------------------------------------

    line = f.readline().strip()


    if line != "RHYME_GROUPS":

        print(
            f"ERROR: Expected RHYME_GROUPS, "
            f"got {line!r}"
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


        # Every word in this group can rhyme
        # with every other word in the group.
        for wid in ids:

            if wid not in rhyme_compatible:

                rhyme_compatible[wid] = set()


            rhyme_compatible[wid].update(
                ids
            )


    # -----------------------------------------------------
    # VOCABULARY
    # -----------------------------------------------------

    id2word = {}
    word2id = {}


    for _ in range(
        vocab_size
    ):

        line = f.readline().strip()


        if not line:
            continue


        parts = line.split(
            " ",
            1
        )


        if len(parts) < 2:
            continue


        try:

            wid = int(
                parts[0]
            )

        except ValueError:

            continue


        word = parts[1]


        id2word[wid] = word
        word2id[word] = wid


# ---------------------------------------------------------
# LOAD BINARY WEIGHTS
# ---------------------------------------------------------
#
# C++ now saves:
#
# weights[position][word][neuron]
#
# followed by:
#
# outputWeights[word][neuron]
#
# ---------------------------------------------------------

print(
    "Loading binary weights..."
)


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


expected_bytes = (
    total_values
    * 4
)


actual_bytes = os.path.getsize(
    WEIGHTS_PATH
)


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
        "ERROR: Binary weight file size does not "
        "match the model dimensions!"
    )

    print(
        f"Expected: {expected_bytes:,} bytes"
    )

    print(
        f"Actual:   {actual_bytes:,} bytes"
    )

    raise SystemExit(1)


# ---------------------------------------------------------
# MEMORY MAP BINARY FILE
# ---------------------------------------------------------
#
# np.fromfile() works, but it loads the complete array
# into RAM first.
#
# memmap() lets the OS handle the file mapping and avoids
# making another giant temporary copy.
#
# ---------------------------------------------------------

all_weights = np.memmap(
    WEIGHTS_PATH,
    dtype=np.float32,
    mode="r",
    shape=(total_values,)
)


# ---------------------------------------------------------
# SPLIT INPUT / OUTPUT WEIGHTS
# ---------------------------------------------------------

input_end = (
    input_values
)


input_data = all_weights[
    :input_end
]


output_data = all_weights[
    input_end:
]


# ---------------------------------------------------------
# RESHAPE INPUT WEIGHTS
# ---------------------------------------------------------
#
# C++ saves:
#
# weights[position][word][neuron]
#
# Shape:
#
# context x word_count x neuron_count
#
# ---------------------------------------------------------

weights = input_data.reshape(
    context,
    word_count,
    neuron_count
)


# ---------------------------------------------------------
# RESHAPE OUTPUT WEIGHTS
# ---------------------------------------------------------
#
# C++ saves:
#
# outputWeights[word][neuron]
#
# Shape:
#
# word_count x neuron_count
#
# ---------------------------------------------------------

out_weights = output_data.reshape(
    word_count,
    neuron_count
)


# ---------------------------------------------------------
# ENSURE NUMPY-FRIENDLY MEMORY LAYOUT
# ---------------------------------------------------------
#
# Do NOT copy these arrays.
#
# The memmap already gives us a valid contiguous view.
#
# Copying here would basically say:
# "hey VPS, can we have another huge RAM problem?"
#
# ---------------------------------------------------------

weights = np.asarray(
    weights,
    dtype=np.float32
)

out_weights = np.asarray(
    out_weights,
    dtype=np.float32
)


print()

print(
    f"Model loaded: "
    f"{len(id2word)} words, "
    f"{neuron_count} neurons, "
    f"{context} context"
)


print(
    f"Rhyme groups loaded: "
    f"{len(rhyme_compatible)} words"
)


print(
    f"Input weights shape:  "
    f"{weights.shape}"
)


print(
    f"Output weights shape: "
    f"{out_weights.shape}"
)


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

CONTEXT = context

MAX_TOKENS = 60

TOP_K = 25


# Random noise is expensive on a small CPU.
USE_NOISE = False


# Keep this relatively small.
RHYME_BOOST = 0.25


# Discourage repetition without completely banning it.
REPEAT_PENALTY = 0.75


# ---------------------------------------------------------
# RANDOM NUMBER GENERATOR
# ---------------------------------------------------------

rng = np.random.default_rng()


# ---------------------------------------------------------
# COMMON WORD FALLBACK
# ---------------------------------------------------------
#
# Tokenizer sorts by frequency, so low IDs tend to be common.
#
# ---------------------------------------------------------

common_ids = np.arange(
    1,
    min(
        500,
        word_count
    ),
    dtype=np.int32
)


if len(common_ids) == 0:

    common_ids = np.arange(
        1,
        min(
            10,
            word_count
        ),
        dtype=np.int32
    )


# ---------------------------------------------------------
# PRECOMPUTED WORD BIAS
# ---------------------------------------------------------
#
# Your old code rebuilt this on EVERY TOKEN.
#
# We only need to build it once.
#
# ---------------------------------------------------------

word_bias = (
    np.arange(
        word_count,
        dtype=np.float32
    )
    * 0.000001
)


word_bias[0] = 0.0


# ---------------------------------------------------------
# GENERATE POEM
# ---------------------------------------------------------

def generate_poem(text):

    # -----------------------------------------------------
    # CLEAN INPUT
    # -----------------------------------------------------

    text = text.strip().lower()


    if not text:
        return ""


    # -----------------------------------------------------
    # REQUEST-LOCAL BUFFERS
    # -----------------------------------------------------
    #
    # These are local to each request so Flask users do not
    # stomp over each other's generation state.
    #
    # -----------------------------------------------------

    hidden = np.zeros(
        neuron_count,
        dtype=np.float32
    )


    scores = np.empty(
        word_count,
        dtype=np.float32
    )


    # Used words still need counts because your original
    # model used a repetition penalty that becomes stronger
    # when a word is repeated more.
    word_usage = {}


    # -----------------------------------------------------
    # SPLIT PROMPT
    # -----------------------------------------------------

    prompt_words = text.split()


    if not prompt_words:
        return ""


    # -----------------------------------------------------
    # FIRST WORD
    # -----------------------------------------------------

    first_word = prompt_words[0]


    first_id = word2id.get(
        first_word,
        0
    )


    # -----------------------------------------------------
    # UNKNOWN WORD
    # -----------------------------------------------------

    if first_id <= 0:

        first_id = int(
            rng.choice(
                common_ids
            )
        )

        first_word = id2word.get(
            first_id,
            "the"
        )


    # -----------------------------------------------------
    # INITIAL OUTPUT
    # -----------------------------------------------------

    generated_words = [
        first_word
    ]


    ids = [
        first_id
    ]


    # -----------------------------------------------------
    # TRACK USED WORDS
    # -----------------------------------------------------

    if (
        0 < first_id
        < word_count
    ):

        word_usage[first_id] = 1


    # -----------------------------------------------------
    # RHYME TARGET
    # -----------------------------------------------------

    # Last actual word from the previous line.
    rhyme_target_id = None


    # -----------------------------------------------------
    # GENERATION LOOP
    # -----------------------------------------------------

    for step in range(
        MAX_TOKENS - 1
    ):

        # -------------------------------------------------
        # GET RECENT CONTEXT
        # -------------------------------------------------

        current_ids = ids[
            -CONTEXT:
        ]


        # -------------------------------------------------
        # BUILD HIDDEN STATE
        # -------------------------------------------------
        #
        # IMPORTANT:
        #
        # The C++ model is POSITION AWARE.
        #
        # We cannot simply do:
        #
        # weights[current_ids]
        #
        # because position matters.
        #
        # We therefore add:
        #
        # weights[0, word]
        # weights[1, word]
        # weights[2, word]
        #
        # etc.
        #
        # -------------------------------------------------

        hidden.fill(0.0)


        for position, wid in enumerate(
            current_ids
        ):

            if (
                0 < wid
                < word_count
                and position < CONTEXT
            ):

                hidden += weights[
                    position,
                    wid
                ]


        # Match the C++ behavior.
        np.clip(
            hidden,
            -1.0,
            1.0,
            out=hidden
        )


        # -------------------------------------------------
        # SCORE EVERY WORD
        # -------------------------------------------------

        np.dot(
            out_weights,
            hidden,
            out=scores
        )


        scores[0] = -np.inf


        # -------------------------------------------------
        # COMMON WORD BIAS
        # -------------------------------------------------

        scores -= word_bias


        # -------------------------------------------------
        # REPETITION PENALTY
        # -------------------------------------------------

        for used_id, count in word_usage.items():

            if (
                0 < used_id
                < word_count
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
                    0 < wid
                    < word_count
                ):

                    scores[wid] += (
                        RHYME_BOOST
                    )


        # -------------------------------------------------
        # OPTIONAL RANDOM NOISE
        # -------------------------------------------------

        if USE_NOISE:

            scores += rng.normal(
                0.0,
                0.001,
                size=word_count
            ).astype(
                np.float32
            )


        # -------------------------------------------------
        # GET TOP K
        # -------------------------------------------------

        k = min(
            TOP_K,
            word_count - 1
        )


        if k <= 0:
            break


        # -------------------------------------------------
        # ARG PARTITION
        # -------------------------------------------------
        #
        # Much faster than sorting the entire vocabulary.
        #
        # -------------------------------------------------

        top_indices = np.argpartition(
            scores,
            -k
        )[-k:]


        # -------------------------------------------------
        # SORT ONLY TOP K
        # -------------------------------------------------

        top_indices = top_indices[
            np.argsort(
                scores[top_indices]
            )[::-1]
        ]


        # -------------------------------------------------
        # VALIDATE CANDIDATES
        # -------------------------------------------------

        valid = []


        for idx in top_indices:

            idx = int(idx)


            if idx <= 0:
                continue


            if not np.isfinite(
                scores[idx]
            ):

                continue


            valid.append(
                idx
            )


        if not valid:
            break


        # -------------------------------------------------
        # PICK FROM TOP 3
        # -------------------------------------------------
        #
        # Keeps generations from becoming completely
        # deterministic while remaining cheap.
        #
        # -------------------------------------------------

        choice_count = min(
            3,
            len(valid)
        )


        best_id = int(
            rng.choice(
                valid[
                    :choice_count
                ]
            )
        )


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
        # NEWLINE
        # -------------------------------------------------

        if next_word == "<NEWLINE>":

            # Don't allow two newlines in a row.
            if (
                generated_words
                and generated_words[-1]
                == "<NEWLINE>"
            ):

                continue


            # Find the final actual word
            # from the current line.
            previous_word_id = None


            for word in reversed(
                generated_words
            ):

                if word == "<NEWLINE>":
                    break


                wid = word2id.get(
                    word
                )


                if wid is not None:

                    previous_word_id = wid

                    break


            # That word becomes the rhyme target
            # for the NEXT line.
            if previous_word_id is not None:

                rhyme_target_id = (
                    previous_word_id
                )


            generated_words.append(
                "<NEWLINE>"
            )


            ids.append(
                best_id
            )


            # IMPORTANT:
            # Don't add <NEWLINE> to word_usage.
            continue


        # -------------------------------------------------
        # NORMAL WORD
        # -------------------------------------------------

        generated_words.append(
            next_word
        )


        ids.append(
            best_id
        )


        word_usage[best_id] = (
            word_usage.get(
                best_id,
                0
            )
            + 1
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
                    " ".join(
                        current_line
                    )
                )


            current_line = []


        else:

            current_line.append(
                word
            )


    if current_line:

        lines.append(
            " ".join(
                current_line
            )
        )


    # -----------------------------------------------------
    # RETURN POEM
    # -----------------------------------------------------

    return "\n".join(
        lines
    )