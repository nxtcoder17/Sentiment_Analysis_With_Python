import requests
from lxml import html

class Zomato:
    def __init__ (self):
        # self.url = url
        self.session = requests.Session()
        self.session.headers.update (self.headers())

    def headers (self):
        headers = {
            'authority': 'www.zomato.com',
            'method': 'GET',
            'path': '/ncr/barbeque-nation-connaught-place-new-delhi/reviews',
            'scheme': 'https',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.9',
            'cookie': 'PHPSESSID=38b76dc51ed7df896f3611f03b5d030e1a821629; csrf=343718b7cfd3db42b87a7b0a2487cf98; zl=en; fbtrack=9fe31920416a951b6c307432a7bd24fe; ak_bmsc=1AD8D475E96FD050D20AEB9B41FD45A368788BEE843500001B054D5DB313E03D~pllP9yRm194YlO2qEihdemQPxMHiMG2eXL9YZaTFkF3TE1UEI5XHa67TTMlFcizJFcKgUIAesHXS7vzLPUNUhK5WKsbFTC55b853rrpvhkKB4cnF89GXnm/v1P647y7C1LWOA4paJ7YsMg1qEfzRPG9WWbXrxj5G3YGOcYEU5igDBz63PS9nID0UYjCfjSLXqAw5Qh56g1gccreRpk29HPNBq6p+en2GeerGKQTL/Sisg=; expab=1; session_id=e565329665323-d816-4857-b152-fc3894120605; dpr=1; G_ENABLED_IDPS=google; lty=subzone; ltv=406; fbcity=1; akaas_WebCanary=1565329525~rv=82~id=919b3bac3c163248d2446661bf510093; bm_sv=292E690CBF0DC6644CCDC029E7D5DEF3~w6FXpccpK2IZimQXaWPlo3XN6crKyOwoAepPX+tQ3IfF33nlskBMBR71og6AZYVUMxaPsGbAXul/ZDdTrgS1Tl7cGE+uiRROFmGT3OAh0AlAzWaI5o58fFHh9IH0d6dONMNdddPFEDItx+pifIet0ApP8Ud56hOA5PNFk06S4IE=',
            'referer': 'https://www.zomato.com/',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36',
            'x-requested-with': 'XMLHttpRequest',

        }
        return headers

    def fetch_review (self, url):
        self.data = { 
                'res_id': '301489',
                'sort': 'reviews-dd',
                }
        r = self.session.post (url, data = data)
        if r.status_code != 200:
            print (f"[{r.status_code}], issue with the URL");
            return;
        source = html.document_fromstring (r.text)
        return source.xpath ("//div[@class='rev-text mbot0']")


if __name__ == '__main__':
    z = Zomato ()
    import sys
    # x = z.fetch_review ('https://www.zomato.com/php/filter_reviews.php')
