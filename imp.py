import warnings
warnings.filterwarnings("ignore")
import networkx as nx

    POS_KEPT = ["ADJ", "NOUN", "PROPN", "VERB"]
class important_list(object):

    def __init__(self,doc = None):
        self.doc = doc

    def increment_edge (graph, node0, node1):
        print("link {} {}".format(node0, node1))

        if graph.has_edge(node0, node1):
            graph[node0][node1]["weight"] += 1.0
        else:
            graph.add_edge(node0, node1, weight=1.0)



    def link_sentence (doc, sent, lemma_graph, seen_lemma):
        visited_tokens = []
        visited_nodes = []

        for i in range(sent.start, sent.end):
            token = doc[i]

            if token.pos_ in POS_KEPT:
                key = (token.lemma_, token.pos_)

                if key not in seen_lemma:
                    seen_lemma[key] = set([token.i])
                else:
                    seen_lemma[key].add(token.i)

                node_id = list(seen_lemma.keys()).index(key)

                if not node_id in lemma_graph:
                    lemma_graph.add_node(node_id)

                print("visit {} {}".format(visited_tokens, visited_nodes))
                print("range {}".format(list(range(len(visited_tokens) - 1, -1, -1))))

                for prev_token in range(len(visited_tokens) - 1, -1, -1):
                    print("prev_tok {} {}".format(prev_token, (token.i - visited_tokens[prev_token])))

                    if (token.i - visited_tokens[prev_token]) <= 3:
                        increment_edge(lemma_graph, node_id, visited_nodes[prev_token])
                    else:
                        break

                print(" -- {} {} {} {} {} {}".format(token.i, token.text, token.lemma_, token.pos_, visited_tokens, visited_nodes))

                visited_tokens.append(token.i)
                visited_nodes.append(node_id)