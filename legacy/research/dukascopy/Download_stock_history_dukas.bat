@echo off
rem Install NODE in  https://nodejs.org/en/download/ accept to create NODE environment variables during installation 
rem the first one to be executed will ask to install duke, you accept  
rem Not all actions are present, some are present and do not work. 

rem CONFIGURE THE DESIRED DATES  -from 2020-01-05 -to 2020-01-07 and the download path 
rem all the information of the command here  https://github.com/Leo4815162342/dukascopy-node 
rem once the .csv files are created they are displayed with sep=',' and the Date column in TimeStamp format. 


set USD_STOCK = ususd
setlocal EnableDelayedExpansion
rem  this are inside @FOLO3 but does not exist in DUKE
rem  UPST MELI RIVN SNOW LYFT ZI GTLB CRWD PINS APPS ASAN AFRM DOCN DDOG SHOP U RBLX CRSR
rem  list of @FOLO3
set "list=goog msft tsla twlo rivn snow adbe uber qcom pypl spot mdb nvda amd adsk amzn nvst hubs epam ttd snap etsy nio"
     

rem Split the list in sub-lists (an array) based on the first letter of the words
for %%w in (%list%) do (
   set "word=%%w"
   echo %%wususd
   npx dukascopy-node -i %%wususd -from 2020-01-05 -to 2020-01-07 -t m15 -f csv  -dir C:\Users\Luis\Downloads --volumes true --flats true
)

exit