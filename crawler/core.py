import requests

from crawler import parser


class PTTSpider():

    headers = {
        'user-agent': ('Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'),
    }

    def get_post(self, link):
        resp = requests.get(link, headers=self.headers)
        return parser.post_content(resp.text)
