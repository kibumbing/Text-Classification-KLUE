from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from bert_model import *

#GPU 사용
device = torch.device("cuda:1")

#BERT 모델, Vocabulary 불러오기 필수
bertmodel, vocab = get_pytorch_kobert_model()

# 토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

# Setting parameters
max_len = 64
batch_size = 32

## 학습 모델 로드
model = torch.load('KoBERT_klue500_model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
model.load_state_dict(torch.load('KoBERT_klue500_model_state_dict.pt'))  # state_dict를 불러

def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append(0)
            elif np.argmax(logits) == 1:
                test_eval.append(1)

        return test_eval[0]

text = ""
f = open("img_2.txt", 'r')
while True:
    line = f.readline()
    if not line:
        break
    use = predict(line)
    print(f"{use} : {line}", end="")
    if use == 0:
        text = text + line
f.close()

file = open('classfication_result.txt', 'w')
file.write(text)
file.close()
print(text)