import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 * Fill in the implementation details of the class DecisionTree
 * using this file. Any methods or secondary classes
 * that you want are fine but we will only interact
 * with those methods in the DecisionTree framework.
 * 
 * You must add code for the 4 methods specified below.
 * 
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {

	public final String POSITIVE = "1"; // the positive label
	public final String NEGATIVE = "2"; // the negative label

	int[] numbers = {4,5,11,5,-1,-1,2}; // # categories per attribute
	int[] start = {1,0,0,1,-1,-1,1}; // starting attribute #

	// attributes for each node (the questions)
	String[] attributeValues = {
			"Status of existing checking account",
			"Credit history",
			"Purpose",
			"Savings account/bonds",
			"Duration in month",
			"Credit amount",
			"foreign worker"
	};

	private List<Integer> numericAttrs = new ArrayList<Integer>(); // list of indices for numeric attributes
	private List<Integer> attrMids = new ArrayList<Integer>(); // list of midpoints for numeric attributes

	private DecTreeNodeImpl root; // the root of the tree

	/**
	 * Answers static questions about decision trees.
	 */
	DecisionTreeImpl() {
		// no code necessary
		// this is void purposefully
	}

	/**
	 * Build a decision tree given only a training set.
	 * 
	 * @param train 
	 * 			the training set
	 */
	DecisionTreeImpl(DataSet train) {
		DecTreeNodeImpl root = initTree(train);
		this.root = root;
	}

	/**
	 * Build a decision tree given a training set then prune it using a tuning set.
	 * 
	 * @param train 
	 * 			the training set
	 * @param tune 
	 * 			the tuning set
	 */
	DecisionTreeImpl(DataSet train, DataSet tune) {
		DecTreeNodeImpl root = initTree(train);
		this.root = root;
		pruneTree(tune, root);
	}

	/**
	 * Finds the midpoint of the maximum and minimum of a certain numerical
	 * attribute to use for thresholding.
	 * @param data the data to find the threshold for
	 * @param attribute the index of the attribute in the instance to find the
	 * midpoint of
	 * @return the value of the midpoint
	 */
	private int findMidpoint(List<Instance> data, int attribute) {

		int max = Integer.MIN_VALUE;
		int min = Integer.MAX_VALUE;

		// look at all instances
		for(Instance inst : data) {
			int value = Integer.parseInt(inst.attributes.get(attribute));
			if(value > max) max = value;
			if(value < min) min = value;
		}

		// return midpoint
		return (max + min) / 2;
	}

	/**
	 * Finds the midpoints for thresholding of the specified attributes
	 * @param data the data to search through
	 * @param attrs the index of the attribute to find the midpoint of
	 */
	private void findThresh(List<Instance> data, int... attrs) {
		
		// reset lists
		numericAttrs.clear();
		attrMids.clear();
		
		for(int attr : attrs) {
			int mid = findMidpoint(data, attr);
			numericAttrs.add(attr);
			attrMids.add(mid);
		}
	}

	/**
	 * Find the attribute value for an instance and question
	 * @param inst the instance to check
	 * @param question the question to look at
	 * @return
	 */
	private String findValue(Instance inst, int question) {
		String value;
		if(question == 4 || question == 5) {
			int discreteVal = Integer.parseInt(inst.attributes.get(question));
			int midVal = attrMids.get(numericAttrs.indexOf(question));
			if(discreteVal <= midVal) value = "A";
			else value = "B";
		} else {
			value = inst.attributes.get(question);
		}
		return value;
	}

	/**
	 * Predicts label based on majority vote for a set of instances
	 * @param instances the instances to find majority for
	 * @return true iff positive is greater than or equal to negative
	 */
	private boolean majorityPositive(List<Instance> instances) {
		int sumPos = 0;
		int sumNeg = 0;

		// find how many are label 1 and 2
		for(Instance inst : instances) {
			if(inst.label.equals(POSITIVE)) sumPos++;
			else sumNeg++;
		}

		return sumPos >= sumNeg;
	}

	/**
	 * Calculates the entropy for a set of instances
	 * @param instances the data to find entropy for
	 * @return the entropy value
	 */
	private double entropy(List<Instance> instances) {

		// if no instances, entropy = 0
		if(instances.size() == 0) return 0;

		int sumPos = 0;
		int sumNeg = 0;

		// find how many are label 1 and 2
		for(Instance inst : instances) {
			if(inst.label.equals(POSITIVE)) sumPos++;
			else sumNeg++;
		}
		int total = sumPos + sumNeg;

		// calculate entropy
		double probPos = (double)sumPos / (double)total;
		double posPart;
		if(probPos != 0) {
			posPart = -1 * probPos * Math.log(probPos) / Math.log(2);
		} else {
			posPart = 0;
		}

		double probNeg = (double)sumNeg / (double)total;
		double negPart;
		if(probNeg != 0) {
			negPart = -1 * probNeg * Math.log(probNeg) / Math.log(2);
		} else {
			negPart = 0;
		}

		return posPart + negPart;
	}

	/**
	 * Builds the full decision tree with the given data
	 * @param train the data to train on
	 * @return the root of the built tree
	 */
	private DecTreeNodeImpl initTree(DataSet train) {

		// find midpoint once
//		findThresh(train.instances, 4, 5);
		
		// initialize question list
		List<Integer> questions = new LinkedList<Integer>();
		for(int i = 0; i < 7; i++) {
			questions.add(i);
		}

		// kick off build tree
		DecTreeNodeImpl root = buildTree(train.instances, questions, "Root");

		return root;
	}

	/**
	 * Recursively builds a decision tree by choosing the next question with
	 * highest entropy
	 * @param examples the instances to build from
	 * @param questions the questions left to ask
	 * @param attribute the value of the current node
	 * @return the root of the finished tree
	 */
	private DecTreeNodeImpl buildTree(List<Instance> examples, List<Integer> questions, String attribute) {
		DecTreeNodeImpl curNode;
		
		// find midpoints each time
		findThresh(examples, 4, 5);

		// if no more instances, choose default
		if(examples.size() == 0) {
			curNode = new DecTreeNodeImpl(POSITIVE, "N/A", attribute, true);
			return curNode;
		}

		// find majority at this node if there are examples
		String label;
		if(majorityPositive(examples)) label = POSITIVE;
		else label = NEGATIVE;

		// if the instances are pure, return pure value
		if(entropy(examples) == 0) {
			curNode = new DecTreeNodeImpl(label, "N/A", attribute, true);
			return curNode;
		}

		// if no more questions, choose default
		if(questions.size() == 0) {
			curNode = new DecTreeNodeImpl(label, "N/A", attribute, true);
			return curNode;
		}

		// find next question
		int nextQuestion = bestQuestion(examples, questions);
		List<List<Instance>> groups = groupByQuestionAttribute(nextQuestion, examples);

		// create new questions list
		List<Integer> updatedQuestions = new LinkedList<Integer>();
		for(int q : questions) {
			if(q != nextQuestion) {
				updatedQuestions.add(q);
			}
		}

		// create new node
		curNode = new DecTreeNodeImpl(label, attributeValues[nextQuestion], attribute, false);
		if(nextQuestion == 4 | nextQuestion == 5) {
			try {
				Instance inst = groups.get(0).get(0);
				int discreteVal = Integer.parseInt(inst.attributes.get(nextQuestion));
				int midVal = attrMids.get(numericAttrs.indexOf(nextQuestion));
				if(discreteVal <= midVal) {
					// value of first list is A, second is B
					curNode.addChild(buildTree(groups.get(0), updatedQuestions, "A"));
					curNode.addChild(buildTree(groups.get(1), updatedQuestions, "B"));
				}
				else {
					// value of first list is B, second is A
					curNode.addChild(buildTree(groups.get(0), updatedQuestions, "B"));
					curNode.addChild(buildTree(groups.get(1), updatedQuestions, "A"));
				}
			} catch(IndexOutOfBoundsException e) {
				Instance inst = groups.get(1).get(0);
				int discreteVal = Integer.parseInt(inst.attributes.get(nextQuestion));
				int midVal = attrMids.get(numericAttrs.indexOf(nextQuestion));
				if(discreteVal <= midVal) {
					// value of second list is A, first is B
					curNode.addChild(buildTree(groups.get(0), updatedQuestions, "B"));
					curNode.addChild(buildTree(groups.get(1), updatedQuestions, "A"));
				}
				else {
					// value of second list is B, first is A
					curNode.addChild(buildTree(groups.get(0), updatedQuestions, "A"));
					curNode.addChild(buildTree(groups.get(1), updatedQuestions, "B"));
				}
			}
		} else {
			for(int i = 0; i < groups.size(); i++) {
				String attrLabel = "A" + (nextQuestion + 1) + (start[nextQuestion] + i);
				curNode.addChild(buildTree(groups.get(i), updatedQuestions, attrLabel));
			}
		}

		return curNode;
	}

	/**
	 * Creates the list for attribute choice names and instances by choice.
	 * @param question the question to create lists for
	 * @param attrChoices the list of attribute label names
	 * @param instancesByAttrChoice the list of instances by choice
	 */
	private void initLists(int question, List<String> attrChoices, List<List<Instance>> instancesByAttrChoice) {

		// clear lists
		attrChoices.clear();
		instancesByAttrChoice.clear();

		if(question != 4 && question != 5) {
			int numChoices = numbers[question];
			int startChoice = start[question];
			for(int i = startChoice; i < startChoice + numChoices; i++) {

				// add choice name to attrChoices
				String name = "A" + (question + 1) + i;
				attrChoices.add(name);

				// add list for each name
				instancesByAttrChoice.add(new ArrayList<Instance>());
			}
		} else {
			// for question 5 and 6, only add A and B
			attrChoices.add("A");
			attrChoices.add("B");
			instancesByAttrChoice.add(new ArrayList<Instance>());
			instancesByAttrChoice.add(new ArrayList<Instance>());
		}

	}

	/**
	 * Returns multiple lists based on their answers to a specified question
	 * @param question the question to ask
	 * @param examples the list of instances to group
	 * @return a list of grouped instances
	 */
	private List<List<Instance>> groupByQuestionAttribute(int question, List<Instance> examples) {

		// split instances by answers
		List<String> choicesInAttr = new ArrayList<String>();
		List<List<Instance>> instancesByAttrChoice = new ArrayList<List<Instance>>();
		initLists(question, choicesInAttr, instancesByAttrChoice);

		for(Instance inst : examples) {

			// find instances value
			String value = findValue(inst, question);

			// put instance into right list
			int attrIndex = choicesInAttr.indexOf(value);
			instancesByAttrChoice.get(attrIndex).add(inst);
		}

		return instancesByAttrChoice;
	}

	/**
	 * Finds the best question for a list of examples based on max information
	 * gain.
	 * @param examples the list of instances
	 * @param questions the available questions
	 * @return the best question
	 */
	private int bestQuestion(List<Instance> examples, List<Integer> questions) {

		List<Double> infoGainVals = new ArrayList<Double>();

		// for each question
		for(int q : questions) {
			// calculate information gain for the question

			// split instances by answers
			List<List<Instance>> instancesByAttrChoice = groupByQuestionAttribute(q, examples);

			// find entropy of new groupings
			double totalEntropy = 0;
			int totalInst = examples.size();
			for(int i = 0; i < instancesByAttrChoice.size(); i++) {
				List<Instance> sameValue = instancesByAttrChoice.get(i);
				int numWithValue = sameValue.size();
				double entropyPart = ((double)numWithValue / (double)totalInst) * entropy(sameValue);
				totalEntropy += entropyPart;
			}

			// information gain = entropy of all examples - conditional entropy
			double infoGain = entropy(examples) - totalEntropy;
			infoGainVals.add(infoGain);
		}

		// return question with max information gain
		double maxGain = -1;
		int maxGainIndex = -1;
		for(int i = 0; i < infoGainVals.size(); i++) {
			/* set new max info gain and index value if less than or equal to 
			 * the previous info gain, less that or equal so that we choose the
			 * question with the largest value
			 */
			if(infoGainVals.get(i) >= maxGain) {
				maxGain = infoGainVals.get(i);
				maxGainIndex = i;
			}
		}

		return questions.get(maxGainIndex);
	}

	/**
	 * Finds all internal nodes of a tree using depth first search
	 * @param node the top node to search down from
	 * @param nodes a list of nodes to be populated
	 */
	private void internalNodes(DecTreeNodeImpl node, List<DecTreeNodeImpl> nodes) {

		// add node if no children
		if(!node.terminal) {
			nodes.add(node);

			// recursively run on children
			for(DecTreeNode child : node.children) {
				internalNodes((DecTreeNodeImpl) child, nodes);
			}
		}
	}
	
	/**
	 * Calculates the accuracy of the tuning set
	 * @param tune the tuning set
	 * @return the accuracy
	 */
	private double calcAccuracy(DataSet tune) {
		
		List<Instance> testInsList = tune.instances;
		String[] results = classify(tune);
		
		// checks
		if(results == null) {
			 System.out.println("Error in calculating accuracy: " +
			 		"You must implement the classify method");
			 System.exit(-1);
		}
		if(testInsList.size() == 0) {
			System.out.println("Error: Size of test set is 0");
			System.exit(-1);
		}
		if(testInsList.size() > results.length) {
			System.out.println("Error: The number of predictions is inconsistant " +
					"with the number of instances in test set, please check it");
			System.exit(-1);
		}
		
		// calculate accuracy
		int correct = 0;
		int total = testInsList.size();
		for(int i = 0; i < testInsList.size(); i ++) {
			if(testInsList.get(i).label.equals(results[i])) correct ++;
		}
		double accuracy = (double)correct / (double)total;
		
		return accuracy;
	}

	/**
	 * Prunes the tree using a greedy algorithm
	 * @param tune the set to prune on
	 * @param root the base of the tree
	 */
	private void pruneTree(DataSet tune, DecTreeNodeImpl root) {
		
		// find internal nodes
		List<DecTreeNodeImpl> internalNodes = new ArrayList<DecTreeNodeImpl>();
		internalNodes(root, internalNodes);

		// find accuracy of current tree
		double origAccuracy = calcAccuracy(tune);
		List<Double> accuracies = new ArrayList<Double>();
		
		// for each internal node
		for(DecTreeNodeImpl curNode : internalNodes) {
		
			// make terminal
			curNode.terminal = true;
		
			// find accuracy and add accuracy to list
			double curAccuracy = calcAccuracy(tune);
			accuracies.add(curAccuracy);
		
			// change back to not terminal
			curNode.terminal = false;
		}
		
		// find best accuracy in list
		int bestAccIndex = -1;
		double bestAccuracy = -1;
		for(int i = 0; i < accuracies.size(); i++) {
			if(accuracies.get(i) > bestAccuracy) {
				bestAccuracy = accuracies.get(i);
				bestAccIndex = i;
			}
		}
		
		/* if the best accuracy is better than the original tree, set as
		 * terminal and remove the children that are supposed to be removed
		 * when making it terminal
		 */
		if(bestAccuracy >= origAccuracy) {
			internalNodes.get(bestAccIndex).terminal = true;
			internalNodes.get(bestAccIndex).children.clear();
			
			// prune again
			pruneTree(tune, root);
		}
		
		// if don't need to prune again, we are done
	}

	@Override
	/**
	 * Evaluates the learned decision tree on a test set.
	 * @return the label predictions for each test instance 
	 * 	according to the order in data set list
	 */
	public String[] classify(DataSet test) {

		String[] classification = new String[test.instances.size()];

		for(int i = 0; i < test.instances.size(); i++) {
			Instance inst = test.instances.get(i);
			DecTreeNodeImpl node = this.root;

			// travel decision tree
			while(!node.terminal) {

				// find value for next choice
				int question = Arrays.asList(attributeValues).indexOf(node.attribute);
				String value = findValue(inst, question);

				// choose next child with same value
				for(DecTreeNode child : node.children) {
					if(child.parentAttributeValue.equals(value)) {
						node = (DecTreeNodeImpl) child;
						break;
					}
				}
			}

			// now at leaf node for current instance, record label
			classification[i] = node.label;
		}

		return classification;
	}

	@Override
	/**
	 * Prints the tree in specified format. It is recommended, but not
	 * necessary, that you use the print method of DecTreeNode.
	 * 
	 * Example:
	 * Root {Existing checking account?}
	 *   A11 (2)
	 *   A12 {Foreign worker?}
	 *     A71 {Credit Amount?}
	 *       A (1)
	 *       B (2)
	 *     A72 (1)
	 *   A13 (1)
	 *   A14 (1)
	 *         
	 */
	public void print() {
		this.root.print(0);
	}

}

