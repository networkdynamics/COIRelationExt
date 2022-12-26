import json
import logging
from collections import Counter

import hydra
import tqdm
from omegaconf import DictConfig


def generate_output(classes, class_string, tp, fn, fp, tn):
    output_strings = []
    for a_c in classes:
        output_strings.append(f'{class_string} {a_c}')
        if (tp[a_c] + tn[a_c] + fp[a_c] + fn[a_c]) == 0:
            recall = 'n/a'
            precision = 'n/a'
            f1 = 'n/a'
            accuracy = 'n/a'
        elif tp[a_c] == 0:
            recall = 0
            precision = 0
            f1 = 0
            accuracy = (tp[a_c] + tn[a_c]) / (tp[a_c] + tn[a_c] + fp[a_c] + fn[a_c])
        else:
            recall = tp[a_c] / (tp[a_c] + fn[a_c])
            precision = tp[a_c] / (tp[a_c] + fp[a_c])
            f1 = 2 * (precision * recall) / (precision + recall)
            accuracy = (tp[a_c] + tn[a_c]) / (tp[a_c] + tn[a_c] + fp[a_c] + fn[a_c])
        output_strings.append(f'Recall: {recall}')
        output_strings.append(f'Precision: {precision}')
        output_strings.append(f'F1: {f1}')
        output_strings.append(f'Acc: {accuracy}')
        output_strings.append(f'Total actual: {tp[a_c] + fn[a_c]}')
        output_strings.append(f'Total prediction: {tp[a_c] + fp[a_c]}\n')
    return output_strings


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    logging.info('Starting eval')
    true_answers = []
    question_classes = []
    logging.info('Reading true answers:')
    with open(cfg.eval.dataset.predict_path) as f:
        for line in tqdm.tqdm(f):
            jsonline = json.loads(line.strip())
            answer = jsonline['answer'].strip().lower()
            if cfg.eval.two_classes:
                if 'yes' in answer:
                    answer = 'yes'
                if 'unknown' in answer:
                    answer = 'no'
            true_answers.append(answer.strip())
            if 'perform analysis in the study? context' in jsonline['question']:
                question_classes.append('analysis')
            elif 'collect data in the study? context' in jsonline['question']:
                question_classes.append('collect data')
            elif 'coordinate the study? context' in jsonline['question']:
                question_classes.append('coordinate')
            elif 'design the study? context' in jsonline['question']:
                question_classes.append('design')
            elif 'fund the study? context' in jsonline['question']:
                question_classes.append('fund')
            elif 'participate in the study? context' in jsonline['question']:
                question_classes.append('participate')
            elif 'review the study? context' in jsonline['question']:
                question_classes.append('review')
            elif 'supply data to the study? context' in jsonline['question']:
                question_classes.append('supply data')
            elif 'supply the study? context' in jsonline['question']:
                question_classes.append('supply')
            elif 'write the study? context' in jsonline['question']:
                question_classes.append('write')
            elif 'support the study? context' in jsonline['question']:
                question_classes.append('support')
    if cfg.eval.no_unknown:
        answer_a_classes = list(set(true_answers) - {'unknown'})
    else:
        answer_a_classes = list(set(true_answers))
    question_a_classes = list(set(question_classes))
    print(Counter(question_classes))
    predict_answers = []
    logging.info('Reading predict answers:')
    with open(cfg.eval.dataset.prediction_path) as f:
        for line in tqdm.tqdm(f):
            answer = line.strip().lower()
            if cfg.eval.two_classes:
                if 'yes' in answer:
                    answer = 'yes'
                elif 'unknown' in answer or 'no' in answer:
                    answer = 'no'
                else:
                    answer = 'no'
            predict_answers.append(answer.strip())
    correct = 0

    tp = {c: 0 for c in answer_a_classes + question_a_classes + ['all'] +
          [f'{x} {y}' for x in answer_a_classes for y in question_a_classes]}
    fn = {c: 0 for c in answer_a_classes + question_a_classes + ['all'] +
          [f'{x} {y}' for x in answer_a_classes for y in question_a_classes]}
    fp = {c: 0 for c in answer_a_classes + question_a_classes + ['all'] +
          [f'{x} {y}' for x in answer_a_classes for y in question_a_classes]}
    tn = {c: 0 for c in answer_a_classes + question_a_classes + ['all'] +
          [f'{x} {y}' for x in answer_a_classes for y in question_a_classes]}
    logging.info('Calculating metrics:')
    assert len(true_answers) == len(predict_answers)
    assert len(true_answers) == len(question_classes), f"{len(true_answers)} & {len(question_classes)}"
    for t, y, q_c in tqdm.tqdm(zip(true_answers, predict_answers, question_classes)):
        if t == y:
            correct += 1
        for a_c in answer_a_classes:
            if y == a_c and t == a_c:
                tp[a_c] += 1
                tp[q_c] += 1
                tp[f'{a_c} {q_c}'] += 1
                tp['all'] += 1
            if y != a_c and t == a_c:
                fn[a_c] += 1
                fn[q_c] += 1
                fn[f'{a_c} {q_c}'] += 1
                fn['all'] += 1
            if y == a_c and t != a_c:
                fp[a_c] += 1
                fp[q_c] += 1
                fp[f'{a_c} {q_c}'] += 1
                fp['all'] += 1
            if y != a_c and t != a_c:
                tn[a_c] += 1
                tn[q_c] += 1
                tn[f'{a_c} {q_c}'] += 1
                tn['all'] += 1
    output_strings = []
    output_strings.extend(generate_output(answer_a_classes, 'Answer Class', tp, fn, fp, tn))
    output_strings.extend(generate_output(question_a_classes, 'Question Class', tp, fn, fp, tn))
    output_strings.extend(generate_output([f'{x} {y}' for x in answer_a_classes for y in question_a_classes], 'Product QxA Class', tp, fn, fp, tn))
    output_strings.extend(generate_output(['all'], 'All', tp, fn, fp, tn))
    print('\n'.join(output_strings))
    open(cfg.eval.dataset.eval_path, 'w').write('\n'.join(output_strings))


if __name__ == '__main__':
    main()
