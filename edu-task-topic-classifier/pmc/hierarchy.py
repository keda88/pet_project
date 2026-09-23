"""Conservative partial-label masks. Pure Python for offline safety tests."""


def relations(schema):
    return {t: set(schema["nodes"][t]["path"][:-1]) for t in schema["nodes"]}


def label_state(targets, schema, labels=None, hierarchical=False):
    labels = schema["labels"] if labels is None else labels
    ancestors = relations(schema)
    targets = set(targets)
    positive = set(targets)
    if hierarchical:
        positive.update(a for t in targets for a in ancestors[t])
    # Descendants under ANY assigned topic remain unknown, unless also positive.
    unknown = {t for t in labels if any(a in ancestors[t] for a in targets)}
    if not hierarchical:
        # An ancestor is not an exact most-specific target, nor a safe negative.
        unknown.update(a for t in targets for a in ancestors[t])
    unknown.difference_update(positive)
    return [int(t in positive) for t in labels], [int(t not in unknown) for t in labels]


def close_topics(topics, schema):
    return set(a for t in topics for a in schema["nodes"][t]["path"])


def most_specific(topics, schema):
    topics = set(topics)
    ancestors = {a for t in topics for a in schema["nodes"][t]["path"][:-1]}
    return sorted(topics - ancestors, key=int)
