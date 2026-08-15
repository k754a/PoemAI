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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "poem_model-0.7m.model"
)


# ---------------------------------------------------------
# OPTIONAL RHYMING
# ---------------------------------------------------------

try:
    import pronouncing
    USE_RHYMES = True
except Exception:
    USE_RHYMES = False


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("Loading model...")

if not os.path.exists(MODEL_PATH):
    print(
        "ERROR: poem_model-0.7m.model not found!"
    )
    raise SystemExit(1)


with open(
    MODEL_PATH,
    encoding="utf-8",
    errors="ignore"
) as f:

    # First three lines contain model information.
    word_count = int(f.readline())
    neuron_count = int(f.readline())
    vocab_size = int(f.readline())


    # -----------------------------------------------------
    # RHYME GROUPS
    # -----------------------------------------------------

    line = f.readline().strip()

    if line != "RHYME_GROUPS":
        print(
            f"ERROR: Expected RHYME_GROUPS, got: {line!r}"
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

        # Every word in this rhyme group can rhyme
        # with every other word in the group.
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

        wid = int(parts[0])
        word = parts[1]

        id2word[wid] = word
        word2id[word] = wid


    # -----------------------------------------------------
    # LOAD WEIGHTS
    # -----------------------------------------------------
    #
    # The model is stored as text, so we load it directly
    # into float32 arrays to keep RAM usage reasonable.
    #
    # -----------------------------------------------------

    def safe_load(rows):

        data = np.zeros(
            (
                rows,
                neuron_count
            ),
            dtype=np.float32
        )

        for i in range(rows):

            line = f.readline()

            if not line:
                break

            try:

                vals = np.fromstring(
                    line,
                    dtype=np.float32,
                    sep=" "
                )

            except Exception:

                vals = np.empty(
                    0,
                    dtype=np.float32
                )

            length = min(
                len(vals),
                neuron_count
            )

            if length > 0:

                data[
                    i,
                    :length
                ] = vals[
                    :length
                ]

        return data


    # Input/embedding weights.
    weights = safe_load(
        word_count
    )

    # Output weights.
    out_weights = safe_load(
        word_count
    )


# ---------------------------------------------------------
# ENSURE NUMPY-FRIENDLY MEMORY LAYOUT
# ---------------------------------------------------------

weights = np.ascontiguousarray(
    weights,
    dtype=np.float32
)

out_weights = np.ascontiguousarray(
    out_weights,
    dtype=np.float32
)


print(
    f"Model loaded: "
    f"{vocab_size} words, "
    f"{neuron_count} neurons"
)

print(
    f"Rhyme words loaded: "
    f"{len(rhyme_compatible)}"
)


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

CONTEXT = 12

# 60 gives the same approximate length as your old version.
MAX_TOKENS = 60

# We only need a few candidates.
TOP_K = 25

# Random noise is expensive on a small CPU.
# Turn this on if you really want it.
USE_NOISE = False

# How strongly rhyme candidates should be preferred.
RHYME_BOOST = 2.0


# ---------------------------------------------------------
# RANDOM NUMBER GENERATOR
# ---------------------------------------------------------

rng = np.random.default_rng()


# ---------------------------------------------------------
# COMMON WORD FALLBACK
# ---------------------------------------------------------
#
# Low IDs are common words because your tokenizer sorts
# vocabulary by frequency.
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
    # These used to be global.
    #
    # That's bad for Flask because two users generating
    # poems at the same time could overwrite each other's
    # buffers.
    #
    # They're tiny compared to the actual model.
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

    used_mask = np.zeros(
        word_count,
        dtype=np.bool_
    )


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
            "?"
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

    used_mask.fill(False)

    if (
        0 < first_id
        < word_count
    ):

        used_mask[first_id] = True


    # -----------------------------------------------------
    # RHYME TARGET
    # -----------------------------------------------------

    # The final word of the previous line.
    #
    # None means the current line does not need to rhyme.
    #
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
        # HIDDEN STATE
        # -------------------------------------------------
        #
        # OLD VERSION:
        #
        # hidden = np.zeros(...)
        #
        # for wid in current_ids:
        #     hidden += weights[wid]
        #
        # That Python loop is slow.
        #
        # NumPy performs the reduction in C instead.
        #
        # -------------------------------------------------

        hidden[:] = np.sum(
            weights[current_ids],
            axis=0,
            dtype=np.float32
        )


        # -------------------------------------------------
        # MATCH C++ MATH
        # -------------------------------------------------

        hidden /= float(
            CONTEXT
        )


        # Keep hidden values between -1 and 1.
        np.clip(
            hidden,
            -1.0,
            1.0,
            out=hidden
        )


        # -------------------------------------------------
        # SCORE EVERY WORD
        # -------------------------------------------------
        #
        # Matrix-vector multiplication.
        #
        # out= allows NumPy to reuse our existing scores
        # array instead of allocating another large one.
        #
        # -------------------------------------------------

        np.dot(
            out_weights,
            hidden,
            out=scores
        )


        # -------------------------------------------------
        # BAN PADDING TOKEN
        # -------------------------------------------------

        scores[0] = -np.inf


        # -------------------------------------------------
        # COMMON WORD BIAS
        # ---------------------------------------------------------

        scores -= word_bias


        # -------------------------------------------------
        # OPTIONAL RANDOM NOISE
        # -------------------------------------------------
        #
        # Disabled by default because creating an entire
        # vocabulary-sized random array for every token is
        # unnecessary CPU work.
        #
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
        # BAN WORDS ALREADY USED
        # -------------------------------------------------

        scores[
            used_mask
        ] = -np.inf


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
                    and not used_mask[wid]
                ):

                    scores[wid] += RHYME_BOOST


        # -------------------------------------------------
        # GET TOP K
        # -------------------------------------------------
        #
        # np.argsort() sorts EVERYTHING.
        #
        # That's wasteful because we only care about the
        # top 25.
        #
        # argpartition() finds the top section much faster.
        #
        # -------------------------------------------------

        k = min(
            TOP_K,
            word_count - 1
        )

        if k <= 0:
            break


        top_indices = np.argpartition(
            scores,
            -k
        )[-k:]


        # -------------------------------------------------
        # SORT ONLY TOP K
        # -------------------------------------------------

        top_indices = top_indices[
            np.argsort(
                scores[
                    top_indices
                ]
            )[::-1]
        ]


        # -------------------------------------------------
        # REMOVE INVALID CANDIDATES
        # -------------------------------------------------

        valid = []

        for idx in top_indices:

            idx = int(idx)

            if idx <= 0:
                continue

            if used_mask[idx]:
                continue

            if not np.isfinite(
                scores[idx]
            ):
                continue

            valid.append(
                idx
            )


        # Nothing left to generate.
        if not valid:
            break


        # -------------------------------------------------
        # CHOOSE FROM TOP 3
        # -------------------------------------------------
        #
        # Picking from the top few prevents every poem from
        # being exactly the same.
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

            # Don't allow two newlines together.
            if (
                generated_words
                and generated_words[-1]
                == "<NEWLINE>"
            ):
                continue

            # Find the final actual word of this line.
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

            # The word ending this line becomes the
            # rhyme target for the NEXT line.
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
            # Don't add <NEWLINE> to used_mask.
            #
            # Otherwise you'd ban ALL future newlines.
            #
            continue


        # -------------------------------------------------
        # ADD WORD
        # -------------------------------------------------

        generated_words.append(
            next_word
        )

        ids.append(
            best_id
        )

        used_mask[
            best_id
        ] = True


    # -----------------------------------------------------
    # RETURN POEM
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

    return "\n".join(
        lines
    )