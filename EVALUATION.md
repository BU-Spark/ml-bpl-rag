# Analysis of RAG Responses

This document analyzes the performance of a Retrieval-Augmented Generation (RAG) system tested on queries categorized into three typologies: specific queries, vague queries, and gibberish or unknown facts. Below are detailed insights into each category, based on the test results and observations.

## Specific Queries

### Examples

- **Query:** Where is boston?  

  **Response:** Boston is located in the state of Massachusetts, in Suffolk County, United States, which is part of North and Central America.  

  **Source:** While the response mentions Suffolk County, the sources only contained this as a keyword and did not explicitly state the connection.  
  
- **Query:** Who was Hitler?  

  **Response:** Sorry, I was unable to find any documents relevant to your query.  

  **Source:** The sources contained information about World War II but did not specifically address Hitler, even if the BPL includes material about him.  

- **Query:** Do you know Ken Follet?  

  **Response:** Based on the provided context, there is no information available about Ken Follet.  

  **Sources:** The sources retrieved were not related to Ken Follett. It appears that the Boston Public Library (BPL) does not have any material about him.  

- **Query:** Do you know the book 1984?  

  **Response:** Sorry, I was unable to find any documents relevant to your query.  

  **Sources:** The retrieved sources were not related to 1984. The BPL database lacks content specific to this book.  

### Notes
The RAG system can easily respond with, "Sorry, I was unable to find any documents relevant to your query," when it lacks specific information to generate a meaningful answer. However, when it does provide answers, they are typically well-constructed and accurate. For the RAG to function effectively, the documents it accesses must also be highly specific. For example, while there may be numerous images of Hitler, the absence of descriptive text accompanying them makes it impossible for the RAG to provide a comprehensive answer. Additionally, since we are working with the Digital Commonwealth rather than a traditional library catalog, it is understandable that there might not be sufficient material on topics like Ken Follett or the book 1984.

## Vague Queries

### Examples
- **Query:** Tell me about neurology.  

  **Response:** Sorry, I was unable to find any documents relevant to your query.  

  **Sources:** While the response indicates no relevant documents, the sources reportedly contained meaningful information about neurology.  

- **Query:** Boston.  

  **Response:** The context provides a comprehensive overview of Boston, highlighting its historical and geographical aspects. It includes a south-east view of the city, a view of the State Capitol, and a depiction of the regional open space system. It also provides historical maps of Boston, including one from the 18th century that shows the city's maritime economy and settlement patterns, and a re-issue of this map from 1835. The context also mentions various landmarks, such as the Ether Monument Public Garden, the site of Anne Hutchinson’s house, and the John Hancock house. The city's history is also touched upon, with references to its settlement by Europeans in 1630 and its previous habitation by Native Americans.  

  **Sources:** The sources are highly relevant and align well with the query, providing detailed historical and geographical information about Boston.  

- **Query:** What happened in 1984?  

  **Response:** Sorry, I was unable to find any documents relevant to your query.  

  **Sources:** All retrieved sources were editions of the same journal from 1984. While temporally related, the content did not address the question directly.  

### Notes
Vague queries pose an interesting challenge for the RAG system. The RAG system often responds with, "Sorry, I was unable to find any documents relevant to your query," even when it retrieves some interesting and relevant sources. It appears that only sufficiently descriptive sources enable the system to generate proper answers. When it does provide an answer, it is typically accurate and on point. However, the database is structured in a way that allows multiple editions of the same journal to exist, leading the RAG to retrieve all editions as meaningful sources, which can be somewhat redundant. To improve the user experience, it would be helpful to limit these cases by reporting only one edition of a journal.

## Gibberish and Unknown Facts

### Examples
- **Query:** Do you know De Bruyne?  

  **Response:** Sorry, I was unable to find any documents relevant to your query.  

  **Sources:** As expected, the response indicates no knowledge of De Bruyne. The retrieved sources were about someone with a similar name.  

- **Query:** Qwghkjhgf.  

  **Response:** Sorry, I was unable to find any documents relevant to your query.  

  **Sources:** The retrieved sources were random, as expected for a nonsensical query.  
  
- **Query:** Hello, I really like dogs, can you give me some dogs please? Thank you!  

  **Response:** Sorry, I was unable to find any documents relevant to your query.  

  **Sources:** The retrieved sources were about dogs.  

### Notes
The system is behaving as expected: when no relevant answer can be generated, the sources retrieved are either random or associated based on similarity in names or words. For instance, a query about "de Bruyne" results in "Les femmes de Bruges," as these are the most similar terms in the database. (Since de Bruyne is a contemporary football player, it is entirely reasonable that the Digital Commonwealth does not contain any information about him.)

## Query in Different Language (Italian)

### Example

- **Query:** Ciao, dove si trova boston?  

  **Response:** Boston si trova negli Stati Uniti, nello stato del Massachusetts. / Sorry, I was unable to find any documents relevant to your query.  

  **Source:** The sources are about Boston, but not as the ones for the same English query / The sources are about Italy, but not related to Boston itself (e.g., Milan or Rome).  

### Notes
Working with another language makes it challenging to receive the same answer consistently. Sometimes, the system provides the correct response (identical to the English version but translated into Italian) and sometimes the default message: "Sorry, I was unable to find any documents relevant to your query." Additionally, the sources retrieved vary from case to case, and the accuracy of the answer seems to depend on the quality and relevance of these sources. It's interesting to see how an Italian query can correspond to sources about Italy and not about the query itself.

## Final Disclaimer
This test was conducted on a partial database. The inability of the RAG system to find specific information may be due to the absence of relevant data in the current product configuration, even though such information might exist in the complete database.
