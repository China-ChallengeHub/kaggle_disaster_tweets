# kaggle_disaster_tweets
kaggle compete

## Competition Description

Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).
But, it’s not always clear whether a person’s words are actually announcing a disaster. 


## evaluation

$$
F_{1}=2 * \frac{\text { precision } * \text { recall }}{\text { precision }+\text { recall }}
$$

|                                        | Acc    | Large  |
| -------------------------------------- | ------ | ------ |
| Transformer + sigmoid                  | 0.8387 | 0.8370 |
| transformer + sigmoid (avg)            | 0.8407 |        |
| transformer+sigmoid + clip_grad        | 0.8408 |        |
| Transformer + softmax                  | 0.8397 |        |
| Transformer + softmax + labelsmoothing | 0.8396 |        |
| Transformer cnn+softmax                | 0.8421 |        |
| Transformer rnn+softmax                | 0.8429 |        |

