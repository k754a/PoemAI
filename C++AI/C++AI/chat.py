import os
import numpy as np

MODEL_PATH = "poem_model.txt"

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("Loading model...")

if not os.path.exists(MODEL_PATH):
    print("ERROR: poem_model.txt not found. Train the C++ model first!")
    exit()

with open(
    MODEL_PATH,
    encoding="utf-8",
    errors="ignore"
) as f:

    word_count = int(f.readline())
    neuron_count = int(f.readline())
    vocab_size = int(f.readline())

    print(f"Word count:   {word_count}")
    print(f"Neuron count: {neuron_count}")
    print(f"Vocab size:   {vocab_size}")

    # -----------------------------------------------------
    # RHYME GROUPS
    # -----------------------------------------------------

    line = f.readline().strip()

    if line != "RHYME_GROUPS":
        print(
            f"ERROR: Expected RHYME_GROUPS, got {line!r}"
        )
        exit()

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

        try:
            wid = int(parts[0])
        except ValueError:
            continue

        word = parts[1]

        id2word[wid] = word
        word2id[word] = wid

    # -----------------------------------------------------
    # SAFE WEIGHT LOADER
    # -----------------------------------------------------

    def safe_load(rows):

        data = []

        for _ in range(rows):

            values = []

            line = f.readline()

            if not line:
                line = ""

            for x in line.split():

                try:
                    value = float(x)

                    if not np.isfinite(value):
                        value = 0.0

                except ValueError:
                    value = 0.0

                values.append(value)

            # Pad
            while len(values) < neuron_count:
                values.append(0.0)

            # Trim
            values = values[:neuron_count]

            data.append(values)

        return np.array(
            data,
            dtype=np.float32
        )

    print("Loading input weights...")
    weights = safe_load(word_count)

    print("Loading output weights...")
    out_weights = safe_load(word_count)


print()
print(
    f"Model loaded: {len(id2word)} words, "
    f"{neuron_count} neurons"
)

print(
    f"Rhyme groups loaded: "
    f"{len(rhyme_compatible)} words"
)

# ---------------------------------------------------------
# COMMON WORD FALLBACK
# ---------------------------------------------------------

# Tokenizer sorts by frequency, so low IDs tend to be common.
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

CONTEXT = 12
MAX_TOKENS = 120
TOP_K = 25

# How strongly rhymes are preferred.
RHYME_BOOST = 2.0

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

    # User's first word becomes the first word.
    first_word = prompt_words[0]

    first_id = word2id.get(
        first_word,
        0
    )

    # Unknown prompt -> random common word
    if first_id <= 0:

        first_id = int(
            np.random.choice(
                common_ids
            )
        )

        first_word = id2word.get(
            first_id,
            "the"
        )

    # -----------------------------------------------------
    # INITIAL STATE
    # -----------------------------------------------------

    ids = [first_id]

    generated_words = [first_word]

    used_ids = {
        first_id
    }

    # Last actual word of the previous line.
    rhyme_target_id = None

    # -----------------------------------------------------
    # GENERATE
    # -----------------------------------------------------

    for step in range(
        MAX_TOKENS - 1
    ):

        # Keep recent context only.
        current_ids = ids[-CONTEXT:]

        # -------------------------------------------------
        # 1. BUILD HIDDEN STATE
        # -------------------------------------------------

        hidden = np.zeros(
            neuron_count,
            dtype=np.float32
        )

        count = 0

        for wid in current_ids:

            if (
                wid > 0
                and wid < word_count
            ):

                hidden += weights[wid]
                count += 1

        # Match the C++ behavior.
        if count > 0:

            hidden /= float(CONTEXT)

            hidden = np.clip(
                hidden,
                -1.0,
                1.0
            )

        # -------------------------------------------------
        # 2. SCORE WORDS
        # -------------------------------------------------

        scores = out_weights @ hidden

        scores = scores.astype(
            np.float32
        )

        # Never choose padding.
        scores[0] = -np.inf

        # Tiny bias toward common words.
        if word_count > 1:

            scores[1:] -= (
                np.arange(
                    1,
                    word_count,
                    dtype=np.float32
                ) * 0.000001
            )

        # Tiny randomness.
        scores += np.random.normal(
            0.0,
            0.001,
            size=scores.shape
        ).astype(np.float32)

        # -------------------------------------------------
        # 3. NEVER REPEAT WORDS
        # -------------------------------------------------

        for used_id in used_ids:

            if (
                used_id > 0
                and used_id < word_count
            ):

                scores[used_id] = -np.inf

        # -------------------------------------------------
        # 4. RHYME BOOST
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
                    and wid not in used_ids
                ):

                    scores[wid] += RHYME_BOOST

        # -------------------------------------------------
        # 5. TOP CANDIDATES
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
        # 6. PICK WORD
        # -------------------------------------------------

        best_id = int(
            np.random.choice(
                top_indices[
                    :min(
                        3,
                        len(top_indices)
                    )
                ]
            )
        )

        next_word = id2word.get(
            best_id,
            "?"
        )

        if next_word == "?":
            break

        # -------------------------------------------------
        # 7. NEWLINE
        # -------------------------------------------------

        if next_word == "<NEWLINE>":

            # Don't allow two newlines in a row.
            if (
                generated_words
                and generated_words[-1] == "<NEWLINE>"
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
                rhyme_target_id = previous_word_id

            generated_words.append(
                "<NEWLINE>"
            )

            ids.append(
                best_id
            )

            # Don't add newline to used_ids.
            continue

        # -------------------------------------------------
        # 8. NORMAL WORD
        # -------------------------------------------------

        generated_words.append(
            next_word
        )

        ids.append(
            best_id
        )

        used_ids.add(
            best_id
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
    # PRINT
    # -----------------------------------------------------

    print()
    print("AI Poem:")

    for line in lines:
        print(line)

    print()