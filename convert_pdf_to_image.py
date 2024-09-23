import argparse
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import os
def convert(pdf_path, output_folder):

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        # 将PDF转换为图片
        images = convert_from_path(pdf_path)

        # 保存每一页为图片
        for i, image in enumerate(images):
            image_path = f"{output_folder}/page_{i + 1}.png"
            image.save(image_path, 'PNG')

        print("PDF已成功转换为图片")

    except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
        print(f"An error occurred: {e}")

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
