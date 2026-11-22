step 1 - pre filtering
step 2 - Hybrid Retireval
step 3 - Cross Encoder Reranking

step 1 -> we narrow down ther vector search space using fixed rules. This rule is applied directyl in the vector store. that way if the user asks for Enropean region we dont want to search us reguion manulas, so we filrter them and narrow the search space. Same can be done 

Step 2 -> Hybrid retrieval means we run two separate paralell searches on the remaing vecotrs. 

1 Dense Search -> uses an embedding model to find chunks that matcgh the meaning of the suestion

2. Sparse Search -> Is a keyword search algorithm. it looks for exact word matches. we need this to match exact codes etc as embedding fall short in matching exact things.

3 Then we use Reciprocal Rank Fusion to merge top scoring results from bioth algorithms 


Then we do Cross Encoder Reranking

The problem is that vector search compares the query and documents separately. A cross encoder takes the top 15 results from RRF and compares the users question side by side with each chunk and calculates a strict relevance score for each chunk. usually between 0 and 1.

Then a threshold is applied. say 0.35. 

Every chunk which has a relwevance score below the threshold gets discardded and only the leftover chunks get sent to the llm

These results then go to the llm

The llm synthesizes the chunks and question and provides a user friendly answer.

Refusal Guardrail -> When the cross encoder discards everything below the threshold, we check how many chunks are left. if there are none, we do not call the llm. and return a message to the user instead oif hallucination