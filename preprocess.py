import json
import logging
from typing import List

import hydra
import tqdm
from omegaconf import DictConfig


def read_raw(path: str, gen_entities: bool) -> List:
    with open(path) as f:
        sources = []
        targets = []
        sources_entity = []
        targets_entity = []
        for line in tqdm.tqdm(f.readlines()):
            data = json.loads(line)
            data['text']
            header = True
            entities = []
            for row in data['summary'].split('<newline>'):
                if header:
                    header = False
                else:
                    cells = row.split('|')
                    cells = cells[1:len(cells)-1]
                    entity_name = cells[0].strip()
                    entities.append(entity_name)
                    sources.append(f'question: did {entity_name} perform analysis in the study? context: {data["text"]}')
                    sources.append(f'question: did {entity_name} collect data in the study? context: {data["text"]}')
                    sources.append(f'question: did {entity_name} coordinate the study? context: {data["text"]}')
                    sources.append(f'question: did {entity_name} design the study? context: {data["text"]}')
                    sources.append(f'question: did {entity_name} fund the study? context: {data["text"]}')
                    sources.append(f'question: did {entity_name} participate in the study? context: {data["text"]}')
                    sources.append(f'question: did {entity_name} review the study? context: {data["text"]}')
                    sources.append(f'question: did {entity_name} supply the study? context: {data["text"]}')
                    sources.append(f'question: did {entity_name} supply data to the study? context: {data["text"]}')
                    sources.append(f'question: did {entity_name} support the study? context: {data["text"]}')
                    sources.append(f'question: did {entity_name} write the study? context: {data["text"]}')
                    targets.append(cells[1])
                    targets.append(cells[2])
                    targets.append(cells[3])
                    targets.append(cells[4])
                    targets.append(cells[5])
                    targets.append(cells[7])
                    targets.append(cells[8])
                    targets.append(cells[9])
                    targets.append(cells[10])
                    targets.append(cells[11])
                    targets.append(cells[12])
            if gen_entities:
                sources_entity.append(f'question: what are all the entities? context: {data["text"]}')
                targets_entity.append(', '.join(entities))
        if gen_entities:
            sources = sources_entity
            targets = targets_entity
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
    train = read_raw(path=cfg.preprocess.dataset.train_path, gen_entities=cfg.preprocess.generate_entities)
    valid = read_raw(path=cfg.preprocess.dataset.valid_path, gen_entities=cfg.preprocess.generate_entities)
    open(cfg.preprocess.dataset.train_output_path, 'w').write('\n'.join(train))
    open(cfg.preprocess.dataset.valid_output_path, 'w').write('\n'.join(valid))

if __name__ == '__main__':
    main()