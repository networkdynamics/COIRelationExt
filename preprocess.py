import json
import logging
from typing import List

import hydra
import tqdm
from omegaconf import DictConfig

def get_questions_re(entity_name: str, context: str, collapse: bool) -> List:
    if collapse:
        sources = [f'question: is the {entity_name} passively involved in the study? context: {context}',
                   f'question: is the {entity_name} actively involved in the study? context: {context}']
    else:
        sources = [f'question: did {entity_name} perform analysis in the study? context: {context}',
                   f'question: did {entity_name} collect data in the study? context: {context}',
                   f'question: did {entity_name} coordinate the study? context: {context}',
                   f'question: did {entity_name} design the study? context: {context}',
                   f'question: did {entity_name} fund the study? context: {context}',
                   f'question: did {entity_name} participate in the study? context: {context}',
                   f'question: did {entity_name} review the study? context: {context}',
                   f'question: did {entity_name} supply the study? context: {context}',
                   f'question: did {entity_name} supply data to the study? context: {context}',
                   f'question: did {entity_name} support the study? context: {context}',
                   f'question: did {entity_name} write the study? context: {context}']
    return sources


def get_target_re(cells: List, collapse: bool) -> List:
    targets = []
    passive_cells = [cells[5], cells[10], cells[3]]
    active_cells = [cells[1], cells[2], cells[4], cells[6],
                     cells[7], cells[8], cells[9], cells[11]]
    if collapse:
        total_active_yes = [1 for c in active_cells if 'yes' in c.strip().lower()]
        total_passive_yes = [1 for c in passive_cells if 'yes' in c.strip().lower()]
        total_active_no = [1 for c in active_cells if 'no' == c.strip().lower()]
        total_passive_no = [1 for c in passive_cells if 'no' == c.strip().lower()]
        if sum(total_passive_yes) > 0:
            targets.append('yes')
        elif sum(total_passive_no) > 0:
            targets.append('no')
        else:
            targets.append('unknown')
        if sum(total_active_yes) > 0:
            targets.append('yes')
        elif sum(total_active_no) > 0:
            targets.append('no')
        else:
            targets.append('unknown')
    else:
        for i in range(1, 12):
            targets.append(cells[i])
    return targets


def get_question_ent(context: str) -> str:
    return f'question: What organizations are involved in the study? context: {context}'


def combine_unlabeled(train_path: str, prediction_path: str) -> List:
    f = open(train_path).readlines()
    fp = open(prediction_path).readlines()
    assert len(f) == len(fp), f'{f} != {fp}'
    results = []
    for question, answer in tqdm.tqdm(zip(f, fp)):
        answer = answer.strip()
        question = json.loads(question.strip())['question']
        results.append(
            json.dumps({
                'question': question,
                'answer': answer
            })
        )
    return results


def read_unlabeled(path: str, ent_path: str = '', gen_entities: bool = False,
                   collapse: bool = False) -> List:
    f = open(path)
    if ent_path != '':
        fe = open(ent_path)
    else:
        fe = None
    sources = []
    targets = []
    if gen_entities:
        for line in tqdm.tqdm(f.readlines()):
            data = json.loads(line)
            sources.append(get_question_ent(data['text']))
            targets.append('')
    else:
        for line, ent_line in tqdm.tqdm(zip(f.readlines(), fe.readlines())):
            entities = ent_line.strip().split('|')
            data = json.loads(line)
            for entity_name in entities:
                entity_name = entity_name.strip()
                if entity_name != '':
                    sources.extend(get_questions_re(entity_name, data["text"], collapse))
                    for i in range(12):
                        targets.append('')
    output_data = []
    for src, tgt in zip(sources, targets):
        output_data.append(json.dumps({
            'question': src,
            'answer': tgt
        }))
    return output_data


def read_raw(path: str, gen_entities: bool = False, collapse: bool = False) -> List:
    with open(path) as f:
        sources = []
        targets = []
        for line in tqdm.tqdm(f.readlines()):
            data = json.loads(line)
            header = True
            entities = []
            for row in data['summary'].split('<newline>'):
                if header:
                    header = False
                else:
                    cells = row.split('|')
                    cells = cells[1:len(cells) - 1]
                    entity_name = cells[0].strip()
                    entities.append(entity_name)
                    if not gen_entities:
                        sources.extend(get_questions_re(entity_name, data["text"], collapse))
                        targets.extend(get_target_re(cells, collapse))

            if gen_entities:
                sources.append(get_question_ent(data['text']))
                targets.append(' | '.join(entities))
        output_data = []
        for src, tgt in zip(sources, targets):
            output_data.append(json.dumps({
                'question': src,
                'answer': tgt
            }))
        return output_data


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    logging.info('Preprocessing start')
    if cfg.preprocess.dataset.combine and cfg.preprocess.dataset.unlabeled:
        predicted_labels = combine_unlabeled(
            train_path=cfg.preprocess.dataset.train_path,
            prediction_path=cfg.preprocess.dataset.prediction_path
        )
        open(cfg.preprocess.dataset.train_output_path, 'w').write('\n'.join(predicted_labels))
    elif cfg.preprocess.dataset.unlabeled:
        unlabeled = read_unlabeled(
            path=cfg.preprocess.dataset.train_path,
            ent_path=cfg.preprocess.dataset.ent_path,
            gen_entities=cfg.preprocess.generate_entities)
        open(cfg.preprocess.dataset.train_output_path, 'w').write('\n'.join(unlabeled))
    else:
        train = read_raw(path=cfg.preprocess.dataset.train_path,
                         gen_entities=cfg.preprocess.generate_entities,
                         collapse=cfg.preprocess.dataset.collapse)
        valid = read_raw(path=cfg.preprocess.dataset.valid_path,
                         gen_entities=cfg.preprocess.generate_entities,
                         collapse=cfg.preprocess.dataset.collapse)
        open(cfg.preprocess.dataset.train_output_path, 'w').write('\n'.join(train))
        open(cfg.preprocess.dataset.valid_output_path, 'w').write('\n'.join(valid))


if __name__ == '__main__':
    main()
