import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
import argparse, os, pickle
import numpy as np
from test import predict_captions

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--accumulation_steps', type=int, default=2)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--vc_features', type=str, default='vc_coco_trainval_2014')
    parser.add_argument('--model_path', type=str, default='vc_coco_trainval_2014')

    args = parser.parse_args()

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, vc_features=args.vc_features, max_detections=50,
                                       load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field=image_field, text_field=text_field, img_root='coco/images/',
                   features_root=args.features_path, ann_root=args.annotation_folder, id_root=args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits

    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))

    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': args.m})
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    data = torch.load(args.model_path)
    model.load_state_dict(data['state_dict'])

    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    ref_caps_train = list(train_dataset.text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    print(len(dict_dataset_test))
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)

    predict_captions(model, dict_dataloader_test, text_field)

