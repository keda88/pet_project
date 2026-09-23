"""Partial hierarchical supervision; pure Python, shared by both models."""
def supervision(row, schema):
    positive = {a for t in row['targets'] for a in schema['nodes'][t]['path']}
    unknown = {t for t in schema['labels'] if t not in positive and
               any(a in schema['nodes'][t]['path'][:-1] for a in row['targets'])}
    return [int(t in positive) for t in schema['labels']], [int(t not in unknown) for t in schema['labels']]


def active_labels(rows, schema):
    seen = {a for r in rows for t in r['targets'] for a in schema['nodes'][t]['path']}
    return [t in seen for t in schema['labels']]


def decode(scores, subject, schema, active, thresholds):
    return [t for j, t in enumerate(schema['labels']) if active[j] and
            schema['nodes'][t]['subject'] == subject and scores[j] >= thresholds[j]]


def frontier(topics, schema):
    return [t for t in topics if not any(t in schema['nodes'][other]['path'][:-1] for other in topics)]
