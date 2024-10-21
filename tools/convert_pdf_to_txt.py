import PyPDF2
import argparse


import os
def convert(pdf_path, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(pdf_path,"rb") as pdf:
        reader=PyPDF2.PdfReader(pdf)
        text = "".join(page.extract_text() for page in reader.pages)
        with open(output_folder+"/pdf2text.txt",'w',encoding = 'utf-8') as txt:
            txt.write(text)

        # 加载 PDF 文件


def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description="Convert PDF to images.")
    
    # 添加参数
    parser.add_argument('--pdf_path', '-i', type=str, help='Path to the input PDF file')
    parser.add_argument('--output_folder', '-o', type=str, help='Path to the output folder for images')
    
    # 解析参数
    args = parser.parse_args()
    
    pdf_path = args.pdf_path
    output_folder = args.output_folder
    convert(pdf_path, output_folder)


if __name__ == "__main__":
    main()
