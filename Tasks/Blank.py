with open('incorrect_predictions.txt', 'r',encoding='UTF-8',errors='ignore') as file:
    # 读取文件内容
    content = file.read()
# 删除所有空格
content_without_spaces = content.replace(" ", "")

# 打开文件以写入方式，并写入无空格的内容
with open('incorrect_predictions.txt', 'w',encoding='UTF-8',errors='ignore') as file:
    file.write(content_without_spaces)