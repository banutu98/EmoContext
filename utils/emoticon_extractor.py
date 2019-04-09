import re


def get_unicode_emoticons_dict():
    emoticons = re.compile('<td>([^.])</td>')
    descriptions = re.compile('<td><b>([a-zA-Z0-9 \-.]+)')
    with open('emoticons_html.txt', encoding='utf-8') as f:
        html = f.read()
    emoticons_res = emoticons.findall(html)
    descriptions_res = descriptions.findall(html)
    descriptions_res = [f'<{desc}>' for desc in descriptions_res]
    return dict(zip(emoticons_res, descriptions_res))


if __name__ == '__main__':
    emoticons = get_unicode_emoticons_dict()
    result_string = 'EMOTICONS_UNICODE = ' + str(emoticons)
    with open('mapping_dict.py', 'a', encoding='utf-8') as f:
        f.write(result_string)
