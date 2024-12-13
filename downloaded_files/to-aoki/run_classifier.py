# Author Toshihiko Aoki
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BertClassifier."""


from mptb import BertClassifier


def classification(
    config_path='config/bert_base.json',
    train_dataset_path='tests/sample_text_class.txt',
    eval_dataset_path='tests/sample_text_class.txt',
    pretrain_path='pretrain/bert.pt',
    tf_pretrain_path=None,
    model_path=None,
    vocab_path='tests/sample_text.vocab',
    sp_model_path='tests/sample_text.model',
    save_dir='classifier/',
    log_dir=None,
    batch_size=2,
    max_pos=128,
    lr=5e-5,
    warmup_proportion=0.1,  # warmup_steps = len(dataset) / batch_size * epoch * warmup_proportion
    epochs=5,
    per_save_epochs=1,
    mode='train',
    label_num=-1,
    balance_weight=False,
    balance_sample=False,
    under_sampling=False,
    under_sampling_cycle=False,
    tokenizer_name='google',
    read_head=False,
    fp16=False,
    task='class',
    device=None,
    quantize=False,
    model_name='bert',
    optimizer='bert',
    encoder_json_path=None,
    vocab_bpe_path=None,
    sw_log_dir='runs'
):

    if under_sampling_cycle:
        under_sampling = True
    if mode == 'train':
        estimator = BertClassifier(
            config_path=config_path,
            max_pos=max_pos,
            vocab_path=vocab_path,
            sp_model_path=sp_model_path,
            pretrain_path=pretrain_path,
            tf_pretrain_path=tf_pretrain_path,
            dataset_path=train_dataset_path,
            header_skip=not read_head,
            label_num=label_num,
            tokenizer_name=tokenizer_name,
            under_sampling=under_sampling,
            fp16=fp16,
            task=task,
            device=device,
            quantize=quantize,
            model_name=model_name,
            encoder_json_path=encoder_json_path,
            vocab_bpe_path=vocab_bpe_path,
            sw_log_dir=sw_log_dir
        )

        estimator.train(
            traing_model_path=model_path,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr, warmup_proportion=warmup_proportion,
            balance_weight=balance_weight,
            balance_sample=balance_sample,
            under_sampling_cycle=under_sampling_cycle,
            save_dir=save_dir,
            per_save_epochs=per_save_epochs,
            optimizer_name=optimizer
        )
        if eval_dataset_path is None:
            return

        eval_data_set = estimator.get_dataset(
            dataset_path=eval_dataset_path, header_skip=not read_head)
        score = estimator.evaluate(dataset=eval_data_set, batch_size=batch_size, log_dir=log_dir)
        print(score)

    else:
        estimator = BertClassifier(
            config_path=config_path,
            max_pos=max_pos,
            vocab_path=vocab_path,
            sp_model_path=sp_model_path,
            model_path=model_path,
            dataset_path=eval_dataset_path,
            header_skip=not read_head,
            label_num=label_num,
            tokenizer_name=tokenizer_name,
            under_sampling=under_sampling,
            fp16=fp16,
            device=device,
            quantize=quantize,
            model_name=model_name,
            encoder_json_path=encoder_json_path,
            vocab_bpe_path=vocab_bpe_path,
        )
        score = estimator.evaluate(batch_size=batch_size, log_dir=log_dir)
        print(score)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='BERT classification.', usage='%(prog)s [options]')
    parser.add_argument('--config_path', help='JSON file path for defines networks.', nargs='?',
                        type=str, default='config/bert_base.json')
    parser.add_argument('--train_dataset_path', help='Training Dataset file (TSV file) path for classification.',
                        nargs='?', type=str,  default=None)
    parser.add_argument('--eval_dataset_path', help='Evaluate Dataset file (TSV file) path for classification.',
                        nargs='?', type=str,  default=None)
    parser.add_argument('--pretrain_path', help='Pre-training PyTorch model path.', nargs='?',
                        type=str, default=None)
    parser.add_argument('--tf_pretrain_path', help='Pre-training TensorFlow(Google) model path.', nargs='?',
                        type=str, default=None)
    parser.add_argument('--model_path', help='Classifier PyTorch model path.', nargs='?',
                        type=str, default=None)
    parser.add_argument('--vocab_path', help='Vocabulary file path for BERT to pre-training.', nargs='?', required=True,
                        type=str)
    parser.add_argument('--sp_model_path', help='Trained SentencePiece model path.', nargs='?',
                        type=str, default=None)
    parser.add_argument('--save_dir', help='Classification model saving directory path.', nargs='?',
                        type=str, default='classifier/')
    parser.add_argument('--log_dir', help='Logging file path.', nargs='?',
                        type=str, default=None)
    parser.add_argument('--batch_size',  help='Batch size', nargs='?',
                        type=int, default=4)
    parser.add_argument('--max_pos', help='The maximum sequence length for BERT (slow as big).', nargs='?',
                        type=int, default=512)
    parser.add_argument('--lr', help='Learning rate', nargs='?',
                        type=float, default=2e-5)
    parser.add_argument('--warmup_steps', help='Warm-up steps proportion.', nargs='?',
                        type=float, default=0.1)
    parser.add_argument('--epochs', help='Epochs', nargs='?',
                        type=int, default=10)
    parser.add_argument('--per_save_epochs', help=
                        'Saving training model timing is the number divided by the epoch number', nargs='?',
                        type=int, default=1)
    parser.add_argument('--mode', help='train or eval', nargs='?',
                        type=str, default='train')
    parser.add_argument('--label_num', help='labels number', nargs='?',
                        type=int, default=-1)
    parser.add_argument('--balance_weight', action='store_true',
                        help='Use automatically adjust weights')
    parser.add_argument('--balance_sample', action='store_true',
                        help='Use automatically adjust samples(random)')
    parser.add_argument('--under_sampling', action='store_true',
                        help='Use automatically adjust under samples')
    parser.add_argument('--under_sampling_cycle', action='store_true',
                        help='Use automatically adjust under samples cycle peer')
    parser.add_argument('--tokenizer', nargs='?', type=str, default='google',
                        help=
                        'Select from the following name groups tokenizer that uses only vocabulary files.(mecab, juman)'
                        )
    parser.add_argument('--read_head', action='store_true',
                        help='Use not include header TSV file')
    parser.add_argument('--fp16', action='store_true',
                        help='Use nVidia fp16 (require apex module)')
    parser.add_argument('--task',  nargs='?', type=str, default='class', help='Target Task (class or choice)')
    parser.add_argument('--device', nargs='?', type=str, default=None, help='Target Runing device name.')
    parser.add_argument('--quantize', action='store_true',
                        help='Use quantized bert (testing),')
    parser.add_argument('--model_name', nargs='?', type=str, default='bert',
                        help=
                        'Select from the following name groups model. (bert, proj, albert)'
                        )
    parser.add_argument('--optimizer', nargs='?', type=str, default='bert',
                        help=
                        'Select from the following name groups optimizer. (bert, adamw, lamb)'
                        )
    parser.add_argument('--encoder_json_path', help='GPT2 encoder JSON file path.', nargs='?', type=str)
    parser.add_argument('--vocab_bpe_path', help='GPT2 encoder bpe file path.', nargs='?', type=str)
    parser.add_argument('--sw_log_dir', help='TensorBoard lgo_dir path.', nargs='?', type=str, default='runs')
    args = parser.parse_args()
    classification(
        config_path=args.config_path,
        train_dataset_path=args.train_dataset_path,
        eval_dataset_path=args.eval_dataset_path,
        pretrain_path= args.pretrain_path,
        tf_pretrain_path=args.tf_pretrain_path,
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        sp_model_path=args.sp_model_path,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        batch_size=args.batch_size,
        max_pos=args.max_pos,
        lr=args.lr,
        warmup_proportion=args.warmup_steps,
        epochs=args.epochs,
        per_save_epochs=args.per_save_epochs,
        mode=args.mode,
        label_num=args.label_num,
        balance_weight=args.balance_weight,
        balance_sample=args.balance_sample,
        under_sampling=args.under_sampling,
        under_sampling_cycle=args.under_sampling_cycle,
        tokenizer_name=args.tokenizer,
        read_head=args.read_head,
        fp16=args.fp16,
        task=args.task,
        device=args.device,
        quantize=args.quantize,
        model_name=args.model_name,
        optimizer=args.optimizer,
        encoder_json_path=args.encoder_json_path,
        vocab_bpe_path=args.vocab_bpe_path,
        sw_log_dir=args.sw_log_dir
    )
