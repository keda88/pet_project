#!/usr/bin/env python3
"""Lossless ID-level normalization. Python 3.9+, standard library only."""
import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def js(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def missing(value):
    return value.strip().lower() in {'', 'nan', 'null', 'none'}


def normalized_text(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', value)).strip()


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert all(None not in row and None not in row.values() for row in rows), 'Malformed CSV row'
        return reader.fieldnames, rows


def topic_ids(value):
    if missing(value):
        return []
    value = value.strip()
    if value.startswith('['):
        values = json.loads(value)
        assert isinstance(values, list)
    else:
        values = re.split(r'[,;|\s]+', value.strip('{}'))
    result = [str(item).strip() for item in values]
    assert all(re.fullmatch(r'\d+', item) for item in result), f'Invalid topics_id: {value}'
    return list(dict.fromkeys(result))


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def semantic_key(row):
    """v1: subject-scoped exact text, or conservative plain-prose numeric template."""
    text = row['task_text_normalized']
    # Markup, formulae, links and digit-adjacent identifiers are deliberately excluded.
    eligible = (len(re.findall(r'[А-Яа-яЁёA-Za-z]+', text)) >= 8
                and len(re.findall(r'[А-Яа-яЁёA-Za-z]', text)) >= 50
                and not re.search(r'[\\<>$]|https?://|www\.', text)
                and not re.search(r'[A-Za-zА-Яа-яЁё_]\d|\d[A-Za-zА-Яа-яЁё_]', text))
    numbers = list(re.finditer(r'\d+(?:[.,]\d+)?', text))
    eligible = eligible and 1 <= len(numbers) <= 12
    # Preserve signs, units, punctuation, case and every non-numeric character.
    key = re.sub(r'\d+(?:[.,]\d+)?', '<NUM>', text) if eligible else text
    kind = 'numeric_template' if eligible else 'exact_text'
    return 'sg_v1_' + digest(js([row['subject_ids'], kind, key])), kind, key


def assign_splits(records, seed):
    strata = defaultdict(lambda: defaultdict(list))
    for row in records:
        strata[row['subject_ids']][row['semantic_group_id']].append(row)
    for subject, groups in sorted(strata.items()):
        total = sum(map(len, groups.values()))
        targets = {'train': total * .8, 'validation': total * .1, 'test': total * .1}
        counts = Counter()
        ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), digest(f'{seed}:{item[0]}')))
        for gid, members in ordered:
            split = max(targets, key=lambda name: (targets[name] - counts[name],
                        digest(f'{seed}:{gid}:{name}')))
            counts[split] += len(members)
            for row in members:
                row['split'] = split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--topics', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    source_hashes = {p.resolve(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.tasks, args.topics)}
    csv.field_size_limit(100_000_000)
    fields, raw = read_csv(args.tasks)
    _, topics_raw = read_csv(args.topics)
    assert fields == ['id', 'task', 'answer', 'topics_id', 'text_of_solution']
    assert all(row['id'].strip() for row in raw), 'Empty task ID'
    topics = {row['id']: row for row in topics_raw}
    assert len(topics) == len(topics_raw), 'Duplicate topic ID'
    paths = {}

    def path_for(tid, visited=()):
        if tid in paths:
            return paths[tid]
        assert tid not in visited, f'Topic cycle: {visited + (tid,)}'
        assert tid in topics, f'Missing hierarchy node: {tid}'
        parent = topics[tid]['parent']
        paths[tid] = ([tid] if parent in {'', '0'} else path_for(parent, visited + (tid,)) + [tid])
        return paths[tid]

    for tid in topics:
        path_for(tid)
    grouped = defaultdict(list)
    for row_number, row in enumerate(raw, 2):
        grouped[row['id']].append((row_number, row))
    records, links = [], []
    max_depth = max(map(len, paths.values()))
    conflicts = Counter()
    for task_id, source_rows in grouped.items():
        variants = {key: list(dict.fromkeys(row[key] for _, row in source_rows)) for key in fields}
        conflict_fields = [key for key in fields if key not in {'id', 'topics_id'} and len(variants[key]) > 1]
        conflicts.update(conflict_fields)
        tids = sorted({tid for _, row in source_rows for tid in topic_ids(row['topics_id'])}, key=int)
        known = [tid for tid in tids if tid in topics]
        unknown = [tid for tid in tids if tid not in topics]
        roots = sorted({paths[tid][0] for tid in known}, key=int)
        specific = [tid for tid in known if not any(tid in paths[other][:-1] for other in known)]
        record = dict(source_rows[0][1])
        record.update(topics_id=js(tids), topic_ids=js(tids),
                      topic_paths=js([[topics[node]['name'] for node in paths[tid]] for tid in known]),
                      topic_path_ids=js([paths[tid] for tid in known]),
                      subject_ids=js(roots), subjects=js([topics[root]['name'] for root in roots]),
                      target_topic_ids=js(specific), unknown_topic_ids=js(unknown),
                      multilabel=int(len(tids) > 1), cross_subject=int(len(roots) > 1),
                      target_multilabel=int(len(specific) > 1),
                      source_row_count=len(source_rows), source_row_numbers=js([n for n, _ in source_rows]),
                      source_records=js([row for _, row in source_rows]),
                      field_conflicts=js(conflict_fields),
                      missing_answer=int(missing(record['answer'])),
                      missing_solution=int(missing(record['text_of_solution'])),
                      task_text_normalized=normalized_text(record['task']),
                      leakage_group_id=digest(normalized_text(record['task'])))
        records.append(record)
        for tid in tids:
            path = paths.get(tid, [])
            occurrence = [n for n, row in source_rows if tid in topic_ids(row['topics_id'])]
            link = dict(task_id=task_id, topic_id=tid, topic_name=topics.get(tid, {}).get('name', ''),
                        subject_id=path[0] if path else '', subject=topics[path[0]]['name'] if path else '',
                        topic_path_ids=js(path), topic_path_names=js([topics[node]['name'] for node in path]),
                        depth=len(path), is_known=int(bool(path)), is_target=int(tid in specific),
                        source_row_count=len(occurrence), source_row_numbers=js(occurrence))
            for level in range(max_depth):
                node = path[level] if level < len(path) else None
                link[f'level_{level}_id'] = node or ''
                link[f'level_{level}_name'] = topics[node]['name'] if node else ''
            links.append(link)
    text_groups = defaultdict(list)
    for row in records:
        text_groups[row['leakage_group_id']].append(row)
    duplicate_text = [group for group in text_groups.values() if len(group) > 1]
    cross_text = [group for group in text_groups.values() if len({s for row in group for s in json.loads(row['subject_ids'])}) > 1]
    label_conflicts = [group for group in duplicate_text if len({row['target_topic_ids'] for row in group}) > 1]
    answer_conflicts = [group for group in duplicate_text if len({row['answer'] for row in group if not missing(row['answer'])}) > 1]
    semantic_groups = defaultdict(list)
    for row in records:
        gid, kind, key = semantic_key(row)
        row.update(semantic_group_id=gid, semantic_group_rule=kind)
        semantic_groups[gid].append(row)
    assign_splits(records, args.seed)
    review = []
    review_groups = {group[0]['leakage_group_id']: group for group in label_conflicts + answer_conflicts}
    for gid, group in sorted(review_groups.items()):
        reasons = []
        if len({r['target_topic_ids'] for r in group}) > 1:
            reasons.append('different_target_topics')
        if len({r['answer'] for r in group if not missing(r['answer'])}) > 1:
            reasons.append('different_nonempty_answers')
        for row in sorted(group, key=lambda r: r['id']):
            review.append(dict(review_group_id=gid, review_reasons=js(reasons),
                group_task_count=len(group), task_id=row['id'], task=row['task'],
                answer=row['answer'], topic_ids=row['topic_ids'],
                target_topic_ids=row['target_topic_ids'], topic_path_ids=row['topic_path_ids'],
                topic_paths=row['topic_paths'], subjects=row['subjects'],
                semantic_group_id=row['semantic_group_id'], split=row['split'],
                source_row_numbers=row['source_row_numbers'], source_records=row['source_records'],
                review_status='pending', reviewer_notes=''))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    assert args.tasks.resolve() not in {(output / 'tasks_normalized_v2.csv').resolve(), (output / 'task_topic_links_v2.csv').resolve()}
    output_names = ['tasks_normalized_v2.csv', 'task_topic_links_v2.csv', 'manual_review_queue_v2.csv', 'analysis_report_v2.md']
    assert not set(source_hashes) & {(output / name).resolve() for name in output_names}
    write_csv(output / 'manual_review_queue_v2.csv', review, list(review[0]) if review else ['review_group_id','task_id','review_status'])
    write_csv(output / 'tasks_normalized_v2.csv', records, list(records[0]))
    write_csv(output / 'task_topic_links_v2.csv', links, list(links[0]))
    # Read back exported files; independently reconcile every original row and task-topic pair.
    _, saved = read_csv(output / 'tasks_normalized_v2.csv')
    _, saved_links = read_csv(output / 'task_topic_links_v2.csv')
    recovered = sorted((n, row) for record in saved for n, row in zip(json.loads(record['source_row_numbers']), json.loads(record['source_records'])))
    assert [row for _, row in recovered] == raw
    assert len(saved) == len(grouped) == len({row['id'] for row in saved})
    expected_pairs = {(row['id'], tid) for row in raw for tid in topic_ids(row['topics_id'])}
    assert expected_pairs == {(row['task_id'], row['topic_id']) for row in saved_links}
    assert len(expected_pairs) == len(saved_links)
    saved_groups = defaultdict(set)
    saved_exact = defaultdict(set)
    for row in saved:
        saved_groups[row['semantic_group_id']].add(row['split'])
        saved_exact[(row['subject_ids'], row['leakage_group_id'])].add(row['split'])
        assert row['semantic_group_id'] == semantic_key(row)[0]
        assert row['split'] in {'train', 'validation', 'test'}
    assert all(len(parts) == 1 for parts in saved_groups.values())
    assert all(len(parts) == 1 for parts in saved_exact.values())
    check_records = [dict(r) for r in reversed(saved)]
    assign_splits(check_records, args.seed)
    assert {r['id']:r['split'] for r in check_records} == {r['id']:r['split'] for r in saved}
    _, saved_review = read_csv(output / 'manual_review_queue_v2.csv')
    assert {(r['review_group_id'],r['task_id']) for r in saved_review} == {(gid,r['id']) for gid,g in review_groups.items() for r in g}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p,h in source_hashes.items())
    subject_counts = Counter(s for row in records for s in json.loads(row['subjects']))
    counts = Counter(tid for row in records for tid in json.loads(row['target_topic_ids']))
    used = {tid for _, tid in expected_pairs}
    exact_duplicates = len(raw) - len({tuple(row[k] for k in fields) for row in raw})
    report = [
        '# Нормализация задач по математике и физике', '',
        '## Источники и воспроизводимость',
        f'- Задачи: `{args.tasks.name}`; SHA-256 `{hashlib.sha256(args.tasks.read_bytes()).hexdigest()}`.',
        f'- Темы: `{args.topics.name}`; SHA-256 `{hashlib.sha256(args.topics.read_bytes()).hexdigest()}`.',
        '- Запуск: `python3 normalize_dataset_v2.py --tasks /path/tasks.csv --topics /path/topics.csv --output-dir /path/output`.',
        '- Нужен Python 3.9+; внешних зависимостей нет. Исходники не изменяются. Порядок строк и JSON детерминирован.',
        '', '## Проверки сохранности и качества',
        f'- Исходных строк: **{len(raw)}**. Уникальных ID: **{len(grouped)}**. Нормализованных строк: **{len(saved)}**.',
        f'- Повторных строк сверх одной на ID: **{len(raw)-len(grouped)}**; ID с несколькими исходными строками: **{sum(len(g)>1 for g in grouped.values())}**.',
        f'- Полных дубликатов всех пяти исходных полей сверх первого: **{exact_duplicates}**.',
        f'- Уникальных пар задача–тема: **{len(links)}**. Все исходные строки восстановлены из экспортированного CSV и побайтово по значениям полей сверены; потеря строк: **0**.',
        f'- Тем в справочнике: **{len(topics)}**, использованных: **{len(used & set(topics))}**, неиспользованных: **{len(set(topics)-used)}**; неизвестных ID тем: **{len(used-set(topics))}**.',
        f'- Циклов, отсутствующих родителей и повторов ID в дереве: **0**. Максимум уровней: **{max_depth}** (level_0 — предмет).',
        f'- Конфликты исходных полей внутри ID: **{dict(conflicts) or "нет"}**.',
        f'- Распределение задач по предметам: **{dict(subject_counts)}**.',
        f'- Несколько исходных тем: **{sum(r["multilabel"] for r in records)}**; несколько наиболее конкретных меток: **{sum(r["target_multilabel"] for r in records)}**.',
        f'- Cross-subject по ID: **{sum(r["cross_subject"] for r in records)}**; по группам одинакового нормализованного текста: **{len(cross_text)}**.',
        '', '| Поле | Пропуски в исходных строках | Пропуски в уникальных задачах |', '|---|---:|---:|']
    for key in fields:
        report.append(f'| {key} | {sum(missing(r[key]) for r in raw)} | {sum(all(missing(r[key]) for _, r in group) for group in grouped.values())} |')
    report += [
        '', 'Пропуском считается пустая строка или NaN/null/none без учёта регистра и пробелов. Исходные значения, включая `NaN`, сохранены как строки.',
        '', '## Дубликаты текста и разметки',
        f'- Групп одинаковых условий у разных ID после NFC и свёртки пробелов: **{len(duplicate_text)}**, задач в них: **{sum(map(len,duplicate_text))}**.',
        f'- Среди них групп с разными наборами целевых тем: **{len(label_conflicts)}**; с разными непустыми ответами: **{len(answer_conflicts)}**.',
        '- Разные метки одинакового текста требуют сверки: это может быть неполная разметка, а не ошибка. Ответы сравниваются как строки; математическая эквивалентность не проверялась.',
        '', '## Проверка cross-subject и примеры',
        'Предмет определяется исключительно корнем пути в topics.csv, без поиска ключевых слов. Реальных примеров с двумя предметами в этих файлах не найдено. Это проверка существующей разметки, а не доказательство отсутствия межпредметного содержания или ошибок разметки. Исходные темы не исправлялись автоматически.',
    ]
    for row in [r for r in records if r['multilabel']][:3]:
        report.append(f'- ID {row["id"]}: {row["task"][:400]} Темы: {row["topic_paths"]}; cross_subject={row["cross_subject"]}.')
    for group in label_conflicts[:3]:
        report.append(f'- Одинаковый текст у ID {", ".join(r["id"] for r in group)}: {group[0]["task"][:250]} Метки: {js({r["id"]:json.loads(r["target_topic_ids"]) for r in group})}.')
    report += [
        '', '## Формат файлов и целевые метки',
        '- `tasks_normalized_v2.csv`: одна строка на ID. `task`, `answer`, `text_of_solution` сохранены без правок. При будущем конфликте берётся первая запись, а конфликт отмечается; все варианты доступны в `source_records`.',
        '- `topics_id` и `topic_ids` — объединённый уникальный список ID в JSON; исходные значения `topics_id` и все пять исходных полей каждой строки сохранены в `source_records`. `source_row_numbers` — порядковые номера CSV-записей с учётом заголовка, не физические строки многострочного текста.',
        '- `topic_path_ids` и `topic_paths` содержат полные пути известных тем; `subjects` и `subject_ids` — их корни. Неизвестные темы сохраняются отдельно и не получают выдуманного пути.',
        '- `multilabel` означает несколько исходных тем; `cross_subject` — несколько корней. Флаги 0/1. `target_topic_ids` убирает назначенного предка, только если назначен его потомок; соседние ветви сохраняются.',
        '- `task_topic_links_v2.csv`: уникальные пары task_id–topic_id, глубина, корень, JSON-путь и отдельные ID/названия каждого уровня. `is_target` отмечает наиболее конкретные назначенные метки.',
        '- Для предмета в текущих данных подходит один класс; архитектуру и схему хранения стоит оставить совместимыми с несколькими предметами. Для разделов рекомендуется иерархическая multilabel-классификация по `target_topic_ids`, с выводом всех предков для оценки.',
        '- Корневая или промежуточная метка без дочерней — частичная разметка: нельзя считать все её дочерние темы отрицательными. Не превращать несколько тем в случайный единственный класс.',
        f'- Различных целевых тем: **{len(counts)}**; с <5 задачами: **{sum(n<5 for n in counts.values())}**; с <20 задачами: **{sum(n<20 for n in counts.values())}**. Для редких тем нужны ручная оценка и/или объединение на уровне предка.',
        '', '## Разбиение без утечек',
        '- Признак модели — условие `task`; при необходимости изображения/формулы после отдельной проверки. `answer`, `text_of_solution`, пути тем, ID и флаги качества не подавать в классификатор условия.',
        '- Обучать по `split` в tasks_normalized_v2.csv; semantic_group_id используется только для разбиения, не как признак модели.',
        '- До обучения сверить очередь ручной проверки. Оценивать subject accuracy, micro/macro F1 по темам и путям с предками; отдельно редкие и частично размеченные задачи. Порог multilabel подбирать на validation; test не использовать для настройки.',
        '', '## Семантические группы и выполненное разбиение',
        '- Правило sg_v1: SHA-256 от JSON [subject_ids, тип правила, ключ текста]. ID устойчив к порядку строк и добавлению других задач; ID зависит от предмета и ключа текста; замена чисел внутри разрешённого шаблона его сохраняет. Ответы и темы ниже предмета в ключ не входят.',
        '- Точное совпадение: NFC и свёртка пробелов, с сохранением регистра и пунктуации. Разные предметы имеют разные группы. Отдельные ID остаются отдельными строками.',
        '- Числовой шаблон: минимум 8 буквенных токенов и 50 букв; от 1 до 12 числовых токенов. Целые и десятичные числа с точкой/запятой заменяются на <NUM>. Все остальные символы, включая знаки и единицы, должны совпасть. При наличии обратной косой черты, <, >, $, http(s)://, www. или цифры рядом с буквой/подчёркиванием применяется только точное совпадение.',
        '- Ограничения: эвристика не доказывает смысловую эквивалентность; числа могут менять сложность и смысл. Она пропускает перефразирование, числа словами, формулы, HTML, изображения и многие варианты записи чисел. Общие слова сами по себе не объединяют задачи. Полное отсутствие семантической утечки за пределами этих правил не гарантируется.',
        f'- Всего групп: **{len(semantic_groups)}**, задач: **{len(records)}**; групп с несколькими ID: **{sum(len(g)>1 for g in semantic_groups.values())}**.',
        f'- Групп с действительно различающимися числовыми вариантами: **{sum(len({r["task_text_normalized"] for r in g})>1 for g in semantic_groups.values())}**; задач в них: **{sum(len(g) for g in semantic_groups.values() if len({r["task_text_normalized"] for r in g})>1)}**; максимальный размер группы: **{max(map(len, semantic_groups.values()))}**.',
        f'- Разбиение 80/10/10, seed={args.seed}. Отдельно внутри каждого набора предметов группы сортируются по убыванию размера, равенство разрешает SHA-256(seed:group_id). Каждая целая группа назначается части с наибольшим дефицитом до целевого числа задач; равенство разрешает SHA-256(seed:group_id:split). Это приближение, точные доли не всегда достижимы. Темы внутри предмета не стратифицируются.',
        '- Проверено после чтения CSV: пересечений semantic_group_id между частями — **0**; пересечений точного текста внутри предмета — **0**. Повторное назначение в обратном порядке строк даёт тот же результат. Хэши обоих исходников до/после совпали.',
        '', '| Предмет | Всего | train | validation | test |', '|---|---:|---:|---:|---:|',
    ]
    for subject in sorted({r['subjects'] for r in records}):
        subset = [r for r in records if r['subjects'] == subject]
        c = Counter(r['split'] for r in subset)
        report.append('| ' + ', '.join(json.loads(subject)) + f' | {len(subset)} | ' + ' | '.join(f'{c[k]} ({c[k]/len(subset):.2%})' for k in ['train','validation','test']) + ' |')
    report += ['', '## Очередь ручной проверки',
        f'- manual_review_queue_v2.csv: **{len(review_groups)}** групп, **{len(review)}** строк задач; включает все **{len(label_conflicts)}** групп с разными наборами целевых тем и все **{len(answer_conflicts)}** групп с различающимися непустыми ответами, без повторного включения общей группы.',
        '- Одна строка на задачу внутри группы. Есть ID группы и задачи, текст, ответ, исходные/целевые темы, полные пути, предмет, split, исходные записи, причина проверки и пустое поле заметок. Статус pending. Ответы сравниваются буквально; эквивалентность не вычисляется. Метки и ответы не исправлялись.',
        '- Все файлы этой поставки имеют суффикс _v2; предыдущая поставка сохранена. Повторный запуск обновляет только результаты _v2. Проверка сохранности восстановила все исходные записи и пары задача–тема.',
    ]
    (output / 'analysis_report_v2.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(js({'input_rows':len(raw),'tasks':len(records),'links':len(links),'cross_subject':sum(r['cross_subject'] for r in records),'text_duplicate_groups':len(duplicate_text),'verified_lossless':True,'output_dir':str(output)}))


if __name__ == '__main__':
    main()
