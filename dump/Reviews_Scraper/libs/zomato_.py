from selenium import webdriver
from selenium.webdriver.chrome.options import Options


class Zomato:
    def __init__ (self):
        options = Options ()
        options.binary_location = "/usr/bin/brave"
        # options.add_argument ('--headless')
        self.driver = webdriver.Chrome(options = options)

    def fetch_review (self, url):
        self.driver.get (url)
        # return self.driver
        while len (self.driver.find_elements_by_xpath ("//div[@class='rev-text mbot0 ']")) == 0:
            pass
        reviews = self.driver.find_elements_by_xpath ("//div[@class='rev-text mbot0 ']")
        counter = 1
        for entry in reviews:
            rating = entry.find_element_by_xpath ("./div").get_attribute ('aria-label')
            rev_text = " ".join (entry.text.split('\n')[1:])
            print (f"[{counter}] {rating}")
            print ('*' * len(rating))
            print ("\t" + rev_text)
            print ()
            counter += 1
            

if __name__ == '__main__':
    import sys
    z = Zomato ()
    z.fetch_review (sys.argv[1])
    # z.fetch_review ('https://www.zomato.com/php/filter_reviews.php')
