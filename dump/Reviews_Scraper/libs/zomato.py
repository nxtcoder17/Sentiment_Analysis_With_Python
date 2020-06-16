from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from selenium.webdriver.common.action_chains import ActionChains

class Zomato:
    def __init__ (self):
        options = Options ()
        options.binary_location = "/usr/bin/brave"
        # options.add_argument ('--headless')
        self.driver = webdriver.Chrome(options = options)

    def fetch_review (self, url):
        self.driver.get (url)
        # Selecting all reviews
        all_reviews = self.driver.find_element_by_xpath ("//*[@id='selectors']/a[2]")
        # all_reviews = self.driver.find_element_by_xpath ("//a[@data-sort='reviews-dd'][@class='item default-section-title everyone empty active selected']")
        print (f"No. of all reviews: {all_reviews.find_element_by_tag_name('span').text}")

        # Load More Button: count
        item = self.driver.find_element_by_xpath ("//span[@class='zs-load-more-count']").text
        print (f"Remaining Load Count: {int(item)}")

        # Load more button
        time.sleep(5)
        btn = self.driver.find_element_by_xpath ("//div[contains('@data-profile_action', 'reviews-top')]")
        btn.click()
        item = self.driver.find_element_by_xpath ("//span[@class='zs-load-more-count']").text
        print (f"Now, Remaining: {int(item)}")


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
