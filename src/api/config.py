class Settings:
    '''
    api settings
    '''
    #文件上传配置
    MAX_FILE_SIZE = 1024 * 1024 * 50    #最大文件大小50M
    ALLOWED_EXTENSIONS = ['.pdf','.docx','html','.md']
    #上传文件存储目录
    UPLOAD_DIR='data/uploads'
#全局配置对象
settings = Settings()