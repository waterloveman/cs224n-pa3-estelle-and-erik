package cs224n.langmodel;

import java.util.Collection;
import java.util.List;

/**
 * Language models assign probabilities to sentences and generate sentences.
 *
 * @author Dan Klein
 */
public interface LanguageModel {

  public void train(Collection<List<String>> trainingSentences);

  public double getSentenceProbability(List<String> sentence);

  public List<String> generateSentence();

}
