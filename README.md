## 训练模型
在终端中输入以下命令来训练模型：
```bash
python scripts/BERT.py
               BERT_history.py
               Transformer.py
               MLWA.py
               slot_gated.py
               slu_baseline.py
```
如果要使用**CRF**可以输入以下命令：
```bash
python scripts/slu_baseline.py --use_crf
```
训练好的模型会保存在`trained_models`路径下。
## 评测模型
在终端中输入以下命令来评测模型：
```bash
python scripts/BERT.py --testing
               Transformer.py 
               MLWA.py
               slot_gated.py
               slu_baseline.py
```
`scripts/BERT.py`的预测结果已经在`data/prediction_BERT.json`中，其他模型的结果会在运行后保存在`data/prediction.json`当中。

对于使用`scripts/BERT_history.py`产生的模型进行预测则运行：
```bash
python scripts/BERT_history_test.py
```
结果会保存在`data/prediction_BERT_history.json`当中。

最好的预测结果在`data/prediction_BERT.json`中。
