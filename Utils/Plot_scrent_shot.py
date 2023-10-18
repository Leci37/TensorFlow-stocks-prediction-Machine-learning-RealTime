import io
from time import sleep
from PIL import Image

from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains




from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from LogRoot.Logging import Logger

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

#https://stackoverflow.com/questions/76724939/there-is-no-such-driver-by-url-https-chromedriver-storage-googleapis-com-lates
driver = webdriver.Chrome(ChromeDriverManager().install())

def __load_drive():
    global driver
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument(f"--window-size=1966,1068")
    options.add_argument(f'--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument("--disable-extensions")
    options.headless = True

    # driver = webdriver.Chrome(options=options)
    driver.set_window_size(1300, 1120)
    driver.minimize_window()# maximize_window()
    Logger.logr.info("Loaded Drive 1200, 800 ChromeDriverManager()")
    return driver

def __remove_elelment_by_xpath(x_path,timeOut ):
    try:
        ele_remove = WebDriverWait(driver, timeOut).until(EC.presence_of_element_located((By.XPATH,x_path )))
        # ele_remove.screenshot("R_"+path_png)
        # ele_remove = driver.find_element(By.CSS_SELECTOR,"body > div: nth - child(10) > div > div > div")
        driver.execute_script("""var element = arguments[0]; element.parentNode.removeChild(element);""", ele_remove)
    except Exception as ex:
         Logger.logr.info("Do not be able to remove xpath element advert banner  " )


def __create_find_data_png_trader_view(path_png):
    ele_screem_shot = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.XPATH,"/html/body/div[3]/div[5]/div/div[1]/div[1]/div[1]/div[2]/div[2]/div/div[2]/div[5]/span[1]/span[1]")))
    ActionChains(driver).move_to_element(ele_screem_shot).click(ele_screem_shot).perform()
    ActionChains(driver).move_to_element(ele_screem_shot).send_keys(Keys.END).perform()
    sleep(0.123)
    list_imge_byte = []
    elements_info = driver.find_elements(By.XPATH,"/html/body/div[3]/div[5]/div/div[1]/div[1]/div[1]/div[2]/div[2]/div/div[2]/div")
    for e in elements_info:
        try:
            list_imge_byte.append(e.screenshot_as_png)
        except:
            pass

    images = [Image.open(io.BytesIO(x)) for x in list_imge_byte]
    widths, heights = zip(*(i.size for i in images))
    total_width = widths[0] * 2  # +1000 #sum(widths)
    max_height = sum(heights) // 2  # +1000
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    y_offset = 0
    for im in images:
        if y_offset >= max_height:
            y_offset = 0
            x_offset = x_offset + widths[0]
        new_im.paste(im, (x_offset, y_offset))
        # x_offset += im.size[0]
        y_offset += im.size[1]
    new_im.save(path_png)

# https://www.urlbox.io/website-screenshots-python
def get_traderview_screem_shot(url, path_png,  will_stadistic_png = True ):
    global driver
    # if driver is None:
    driver = __load_drive()

    driver.get(url)
    Logger.logr.info(" driver.get(url): "+url)

    path_imgs_tech = None
    path_imgs_finan = None
    try:
        __remove_elelment_by_xpath("/html/body/div[7]/div/div/div", timeOut=18)#banner de la cokkie
        __remove_elelment_by_xpath('//*[@id="overlap-manager-root"]', timeOut=2) #banner de unete a la comunidad

        ele_screem_shot = WebDriverWait(driver, 18).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'body > div.js-rootresizer__contents.layout-with-border-radius > div.layout__area--center')))
        is_down_image = ele_screem_shot.screenshot(path_png+"_TRAVIEW_tech.png")
        if is_down_image:
            path_imgs_tech = path_png+"_TRAVIEW_tech.png"
        if will_stadistic_png:
            __create_find_data_png_trader_view(path_png + "_TRAVIEW_finan.png")
            path_imgs_finan = path_png + "_TRAVIEW_finan.png"
        Logger.logr.info("Imagenes Paths: "+ path_png+"_TRAVIEW_tech.png " + path_png + "_TRAVIEW_finan.png")
        return path_imgs_tech, path_imgs_finan
    except TimeoutException:
        Logger.logr.info("Loading took too much time!")
    except Exception as ex:
        Logger.logr.info("Exception: " + str(ex))

    return path_imgs_tech, path_imgs_finan










# 5
# BTC_path = "https://es.tradingview.com/chart/x0nIfdrH/?symbol=BITSTAMP%3ABTCUSD"
# UNITY = "https://www.tradingview.com/chart/x0nIfdrH/?symbol=NYSE%3AU"
# path_imgs_tech, path_imgs_finan = get_traderview_screem_shot('https://www.tradingview.com/chart/x0nIfdrH/?symbol=NASDAQ%3AMELI', 'MELI')
# # s.capture('https://www.tradingview.com/chart/x0nIfdrH/?symbol=NASDAQ%3AMELI', 'MELI.png')
# get_traderview_screem_shot('https://www.tradingview.com/chart/x0nIfdrH/?symbol=NASDAQ%3AMDB', 'MDB')
# get_traderview_screem_shot(BTC_path, 'BTC-USD')
# get_traderview_screem_shot(UNITY, 'U')
# driver.quit()
# sys.exit(app.exec_())
