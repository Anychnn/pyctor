# pyctor
中文纠错工具

## example
```
corrector = NcnnCorrector()

input_text="你好,很高姓见到你!"
pred_text, list_random_text, offset_mapping = corrector.correct(
                random_text)
print("".join(pred_text))
```