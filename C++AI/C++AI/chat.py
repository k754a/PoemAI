import os
import numpy as np

MODEL_PATH = "poem_model.txt"
WEIGHTS_PATH = "poem_weights.bin"

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("Loading model...")

if not os.path.exists(MODEL_PATH):
    print("ERROR: poem_model.txt not found. Train the C++ model first!")
    raise SystemExit

if not os.path.exists(WEIGHTS_PATH):
    print("ERROR: poem_weights.bin not found. Train the C++ model first!")
    raise SystemExit

with open(
    MODEL_PATH,
    encoding="utf-8",
    errors="ignore"
) as f:

    word_count = int(f.readline())
    neuron_count = int(f.readline())
    context = int(f.readline())
    vocab_size = int(f.readline())

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
        raise SystemExit

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

        parts = line.split(" ", 1)

        if len(parts) < 2:
            continue

        # Smart parsing: try ID WORD first, fall back to WORD ID
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
# LOAD BINARY WEIGHTS
# ---------------------------------------------------------

print("Loading binary weights...")

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
        "ERROR: Binary weight file size does not match "
        "the model dimensions!"
    )
    print(f"Expected: {expected_bytes:,} bytes")
    print(f"Actual:   {actual_bytes:,} bytes")
    raise SystemExit


# ---------------------------------------------------------
# READ BINARY FILE
# ---------------------------------------------------------

all_weights = np.fromfile(
    WEIGHTS_PATH,
    dtype=np.float32
)

if all_weights.size != total_values:
    print("ERROR: Loaded the wrong number of weight values!")
    print(f"Expected: {total_values:,}")
    print(f"Loaded:   {all_weights.size:,}")
    raise SystemExit


# ---------------------------------------------------------
# SPLIT INPUT / OUTPUT WEIGHTS
# ---------------------------------------------------------

input_end = input_values
input_data = all_weights[:input_end]
output_data = all_weights[input_end:]

weights = input_data.reshape(
    context,
    word_count,
    neuron_count
)

out_weights = output_data.reshape(
    word_count,
    neuron_count
)

del all_weights
del input_data
del output_data

print()
print(
    f"Model loaded: {len(id2word)} words, "
    f"{neuron_count} neurons, "
    f"{context} context"
)
print(f"Rhyme groups loaded: {len(rhyme_compatible)} words")
print(f"Input weights shape:  {weights.shape}")
print(f"Output weights shape: {out_weights.shape}")

# ---------------------------------------------------------
# COMMON WORD FALLBACK
# ---------------------------------------------------------

common_ids = list(
    range(
        1,
        min(500, word_count)
    )
)

if not common_ids:
    common_ids = list(
        range(
            1,
            min(10, word_count)
        )
    )

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

CONTEXT = context
MAX_TOKENS = 140
TOP_K = 30
RHYME_BOOST = 0.25
REPEAT_PENALTY = 0.75

print()
print("--- AI Poet ---")
print("Type 'exit' to stop.")
print()

# ---------------------------------------------------------
# GENERATION LOOP
# ---------------------------------------------------------

while True:

    try:
        text = input("> ").strip().lower()

    except (
        KeyboardInterrupt,
        EOFError
    ):
        print("\nbye.")
        break

    if text == "exit":
        break

    if not text:
        continue

    # -----------------------------------------------------
    # PROMPT
    # -----------------------------------------------------

    prompt_words = text.split()

    if not prompt_words:
        continue

    first_word = prompt_words[0]
    first_id = word2id.get(first_word, 0)

    if first_id <= 0:
        first_id = int(np.random.choice(common_ids))
        first_word = id2word.get(first_id, "the")

    # -----------------------------------------------------
    # INITIAL STATE
    # -----------------------------------------------------

    ids = [first_id]
    generated_words = [first_word]
    word_usage = {first_id: 1}
    rhyme_target_id = None

    # -----------------------------------------------------
    # GENERATE
    # -----------------------------------------------------

    for step in range(MAX_TOKENS - 1):

        current_ids = ids[-CONTEXT:]

        # -------------------------------------------------
        # BUILD HIDDEN STATE
        # -------------------------------------------------

        hidden = np.zeros(
            neuron_count,
            dtype=np.float32
        )

        for position, wid in enumerate(current_ids):
            if (
                wid > 0
                and wid < word_count
                and position < CONTEXT
            ):
                hidden += weights[position, wid]

        hidden = np.clip(hidden, -1.0, 1.0)

        # -------------------------------------------------
        # SCORE WORDS
        # -------------------------------------------------

        scores = out_weights @ hidden

        # Clean up any NaNs or Infs from the C++ training
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0e10, neginf=-1.0e10)
        
        scores = scores.astype(np.float32)

        scores[0] = -np.inf

        if word_count > 1:
            scores[1:] -= (
                np.arange(
                    1,
                    word_count,
                    dtype=np.float32
                )
                * 0.000001
            )

        # -------------------------------------------------
        # REPETITION PENALTY
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
                set()
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

        scores += np.random.normal(
            0.0,
            0.001,
            size=scores.shape
        ).astype(np.float32)

        # -------------------------------------------------
        # TOP CANDIDATES
        # -------------------------------------------------

        candidate_count = min(
            TOP_K,
            word_count - 1
        )

        if candidate_count <= 0:
            break

        top_indices = np.argsort(
            scores
        )[-candidate_count:][::-1]

        top_indices = [
            int(i)
            for i in top_indices
            if (
                i > 0
                and i < word_count
                and np.isfinite(scores[i])
            )
        ]

        if not top_indices:
            break

        # -------------------------------------------------
        # PICK WORD
        # -------------------------------------------------

        best_id = int(top_indices[0])
        next_word = id2word.get(best_id, "?")

        if next_word == "?":
            break

        # -------------------------------------------------
        # NEWLINE
        # -------------------------------------------------

        if next_word == "<NEWLINE>":

            if (
                generated_words
                and generated_words[-1] == "<NEWLINE>"
            ):
                continue

            previous_word_id = None

            for word in reversed(generated_words):

                if word == "<NEWLINE>":
                    break

                wid = word2id.get(word)

                if wid is not None:
                    previous_word_id = wid
                    break

            if previous_word_id is not None:
                rhyme_target_id = previous_word_id

            generated_words.append("<NEWLINE>")
            ids.append(best_id)
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

    # -----------------------------------------------------
    # PRINT
    # -----------------------------------------------------

    print()
    print("AI Poem:")

    for line in lines:
        print(line)

    print()