# coding='UTF-8'
import pandas as pd


if __name__ == '__main__':
    base_file = './Article_labels.csv'
    exist_file = './pages/plain_txt_name_index.csv'

    # 找出在标签表中，已经通过下载拿下来的文件list
    base_data = pd.read_csv(base_file)
    exist_data = pd.read_csv(exist_file)

    FA_data = exist_data[exist_data['flaw_type']=='FA']
    FA_data = FA_data.reset_index(drop=True)

    exist_names = exist_data['article_names'].tolist()
    # 构建存储下载页面的index，供后期切片使用
    indexs =[]

    for index in range(len(base_data)):
        article_name = base_data['article_name'][index].replace(' ', '_')
        if article_name in exist_names:
            indexs.append(index)

    print('exist count is:', len(indexs))
    print('total:', len(base_data), 'exist total:', len(exist_data))
    # 将上面的带标签的文档单独拿出来，并且清洗其中的文件命名规范
    cleaned_data = base_data.iloc[indexs,:7].reset_index(drop=True)
    for index in range(len(cleaned_data)):
        cleaned_data.iloc[index,0] = cleaned_data['article_name'][index].replace(' ', '_')


    # 将带标签的文档和不带标签的FA类文档融合成一张表
    for index in range(len(FA_data)):
        line = pd.Series({'article_name' : FA_data['article_names'][index],'no_footnotes' : 0,'primary_sources' : 0,'refimprove' : 0,'original_research' : 0,'advert' : 0,'notability' : 0})
        cleaned_data = cleaned_data.append(line, ignore_index=True)
    print('FA added exist count is:', len(cleaned_data))
    exist_data = exist_data.iloc[:,:2]
    # 将标签表和文件名表融合，最终的表显示标签、文件名和文章名字
    merged_file = pd.merge(cleaned_data,
                           exist_data,
                           left_on='article_name',
                           right_on='article_names',
                           how='outer')
    print(merged_file.head())

    merged_file.to_csv('./pages/plain_txt_name_index_with_labels.csv')


