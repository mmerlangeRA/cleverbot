import urllib

def get_file_extension(url:str)->str:
    file_extension = urllib.parse.urlparse(url).path.split('.')[-1].lower()
    return file_extension