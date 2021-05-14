import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, experiment,dataset ,train_path,dev_path,test_path,class_list,see):
        self.model_name = 'bert'+dataset
        self.train_path = train_path                               # 训练集
        self.dev_path = dev_path                                  # 验证集
        self.test_path = test_path                                 # 测试集
        self.class_list = class_list                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name+str(see) +experiment+ '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 32                                           # mini-batch大小
        self.pad_size = 128                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = 'bert-base-chinese'  # 预训练模型
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        self.attention_probs_dropout_prob= 0.1
        self.directionality = "bidi"
        self.hidden_act="gelu"
        self.hidden_dropout_prob= 0.1
        self.hidden_size= 768
        self.initializer_range= 0.02
        self.intermediate_size= 3072
        self.layer_norm_eps= 1e-12
        self.max_position_embeddings= 512
        self.model_type= "bert"
        self.num_attention_heads= 12
        self.num_hidden_layers= 12
        self.pad_token_id= 0
        self.pooler_fc_size= 768
        self.pooler_num_attention_heads= 12
        self.pooler_num_fc_layers= 3
        self.pooler_size_per_head= 128
        self.pooler_type= "first_token_transform"
        self.type_vocab_size= 2
        self.vocab_size=21128

        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out