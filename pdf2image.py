import pdf2image
from pdb import set_trace

# from pdf2image import pdf2image 

# 设置PDF文件的路径和输出路径
pdf_path = 'input.pdf'
output_folder = 'pdf_image_folder'

# 将PDF转换为图片
set_trace()
images = convert_from_path(pdf_path)

# 保存每一页为图片
for i, image in enumerate(images):
    image_path = f"{output_folder}/page_{i + 1}.png"
    image.save(image_path, 'PNG')

print("PDF已成功转换为图片")