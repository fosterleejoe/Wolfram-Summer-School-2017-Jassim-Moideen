## Title

Churn Classification for Mobile Telecom CDR data using a Neural Network in Wolfram Language.

## Objective

Communication service providers produce large volumes of calling data records (CDR) every minute day in and day out with details like inbound calls, outbound calls, dropped calls, abandoned calls and unanswered calls.The data can include call detail data, network and customer data based on the configuration of the respective systems. Churn in telecommunication industry happens when the customers leaving the current brand and moving to another telecom company. With  the increasing number of churns, it becomes the operator's process to retain the profitable customers known as churn management. In communication service provider (CSP) industry each company provides the customers with huge incentives to lure them to switch to their.

The approach here is to aggregate the data for the required analysis and classify potential customers who might churn. The outcome can be used for various business use cases like customer profiling, targeted marketing, product design, network fault isolation and fraud detection. With the impending risk of OTT VoIP cannibalization, the clear understanding of their customers behaviour is the vital for the business to maintain their steady revenue stream.

## Data Description

The description of the input data fields are as below :-

	phone : (discrete) number to uniquely identify a subscriber
	account length: (continuous)  tenure of the customer with the brand
	number vmail messages: (continuous)  - number of voice mail messages
	total day minutes: (continuous) - total number of minutes of mobile usage during the day time hours
	total day calls: (continuous) - total number of calls made during the day time hours
	total day charge: (continuous) - total amount of incurred charges for the usage during the day time hours
	total eve minutes: (continuous)  - total number of minutes of mobile usage during the evening time hours
	total eve calls: (continuous)  - total number of calls made during the evening time hours
	total eve charge: (continuous).- total amount of incurred charges for the usage during the evening time hours
	total night minutes: (continuous)  - total number of minutes of mobile usage during the night time hours
	total night calls: (continuous)  - total number of calls made during the night time hours
	total night charge: (continuous) - total amount of incurred charges for the usage during the night time hours
	total intl minutes: (continuous) - total number of minutes mobile usage for international outgoing calls
	total intl calls: (continuous) - total number of international outgoing calls
	total intl charge: (continuous)- total amount of incurred charges for the international outing calls
	number customer service calls: (continuous)- total number of customer support calls made
	
"Phone" should not be used as part of the training, since it has no predictive value. 

The last column "Churn" is the classification (True, False)
