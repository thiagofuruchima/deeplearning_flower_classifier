import argparse
from dnn_image_classifier import flower_classifier

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('path_to_image',
                    action="store",
                    help='Path to the image to be classified')

    parser.add_argument('path_to_model',
                    action="store",
                    help='Path to Keras Model to use to classify the image')
    
    parser.add_argument('--top_k', action='store',
                    dest='top_k',
                    default=5,
                    type=int,
                    help='Return the top KK most likely classes')
    
    parser.add_argument('--category_names', action='store',
                    dest='category_names',
                    default=[],
                    help='Path to a JSON file mapping labels to flower names')
                    
    args = parser.parse_args()
       
    model = flower_classifier.load_model(args.path_to_model)
    
    probs, labels = flower_classifier.predict(args.path_to_image, model, args.top_k)
    
    if (len(args.category_names)):
        result = flower_classifier.get_flower_names(labels, args.category_names)
    else:
        result = labels
    
    print (result)
        
if __name__ == '__main__':
    main()