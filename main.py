from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
from datasets import Dataset
import pandas as pd

#read the csv file with all the quotes
data = pd.read_csv("C:\\Users\\Owner\\OneDrive\\Desktop\\CSEN166\\LAB3\\quotes.csv")

# split the categories list to determine which categories are relevant
# categories_list = data['categories'].str.split(',').explode()
# uni = categories_list.unique()
# print(uni)
# uni = np.array(uni)

#Tags to include and exclude from the training set
include = ['Failure', 'Succeed', 'Experience' ,'Success' ,'Great' ,'Greatest', 'Want','Focus',
            'Support','Learning','Growth', 'Value' ,'Mistakes', 'Risk' ,'Learn','Together',
            'Believe', 'Grow','Motivational', 'I Am', 'Opportunity' ,'Win','Will', 'Best', 'Effort', 'Trying', 'Fear', 'Positive',
            'Future', 'Faith','Try', 'Hope', 'Dreams', 'Sports', 'Game' ,'Inspirational','Passion','Comfort', 'Place' 'Attitude', 'Personality', 
            'Success Is', 'Better', 'Courage','Goals','Fall', 'Stand', 'Pain', 'Goal', 'Strong', 'Successful', 'Hard Work', 'Team','Easy', 'Struggle', 'Strength', 
            'Process', 'Problem']
exclude  = ['America', 'Gift', 'Thought', 'Job', 'My Life', 'Honesty', 'New', 'Morning', 'Day', 'Teacher', 
            'Wise', 'Husband', 'Financial', 'Animal', 'Thinking', 'Mom', 'Money', 'Long', 'Mother', 'School', 
            'Knowledge', 'Food', 'Beauty', 'Relationship', 'Hate', 'Diet', 'Law', 'Live', 'Fool', 
            'Health', 'Kind', 'Time', 'Cooking', 'Justice', 'Friend', 'Society', 'Science', 
            'American', 'Love Is', 'Wife', 'Two', 'Good Food', 'Treat', 'Giving', 'Funny', 'Survival', 'Humor', 
            'Road', 'Light', 'Her', 'Politics', 'Step', 'Death', 'Others', 'Happiness', 'Education', 'Real', 
            'Writing', 'Words', 'Man', 'Character', 'Technology', 'Women', 'Parents', 'Loving', 'Living', 'Work', 'Wisdom', 'World', 'Life', 
            'Men', 'Enjoy Life', 'Small', 'Government', 'PlaceAttitude', 'Love', 'Just', 'Nature', 'Care', 'Happy', 'Music', 
            'Romantic', 'Anniversary', 'Art', 'Freedom', 'Flower', 'Home', 'Power', 'Place', 
            'Culture', 'Truth', 'True Love', 'Joy', 'Equality', 'Young', 'Business', 'History', 'Community', 
            'Language', 'Water', 'Eat', 'Deep', 'Die', 'Find', 'Alone', 'True', 'Marriage', 'Peace', 'Look', 'Smile', 'Person', 
            'Woman', 'Children', 'Travel', 'Finance', 'God', 'Creativity', 'Night', 'Good', 'Friendship', 'Beautiful', 'Book', 'Body', 'Human', 
            'Age', 'Family', 'Gay', 'Child', 'Life Is A', 'Enjoy', 'Father', 'Eyes', 'Looking']

#remove quotations and symbols from the quote ('text') column
def preprocess(text):
    text = text.strip('"\'')
    text = ' '.join(text.split())
    if not text.endswith(('.', '!', '?')):
        text +='.'
    return text

#rewrite 'text' columns with the preprocessed text
data['text'] = data['text'].apply(preprocess)

#filter the data based on include and exclude categories
filtered = data[data['categories'].fillna('').apply(lambda cat: any(word in [c.strip() for c in cat.split(',')] for word in include) 
                                         and not any(word in [c.strip() for c in cat.split(',')] for word in exclude))]


#classify tags to focuses
focus_mapping = {
    'Courage': ['Courage', 'Fear', 'Risk', 'Failure', 'Experience', 'Try', 'Trying', 'Faith', 'Hope', 'Dreams', 'Fall', 'Stand', 'Pain', 'Struggle', 'Strength', 'Believe', 'Motivational', 'Inspirational', 'Passion'],
    'Perseverance': ['Struggle', 'Try', 'Never', 'Fall', 'Failure', 'Trying', 'Effort', 'Pain', 'Hard Work', 'Process', 'Strength', 'Believe', 'Motivational', 'Inspirational', 'Passion'],
    'Goal Setting': ['Goals', 'Goal', 'Dreams', 'Purpose', 'Success', 'Successful', 'Win', 'Best', 'Future', 'Opportunity', 'Believe', 'Motivational', 'Inspirational', 'Passion'],
    'Confidence': ['Believe', 'Trust', 'Strong', 'Strength', 'Believe', 'I Am', 'Positive', 'Attitude', 'Personality', 'Faith', 'Motivational', 'Inspirational'],
    'Growth Mindset': ['Learn', 'Learning', 'Grow', 'Growth', 'Change', 'Mistakes', 'Experience', 'Value', 'Process', 'Comfort', 'Motivational', 'Inspirational'],
    'Resilience': ['Mistakes', 'Failure', 'Success', 'Positive', 'Struggle', 'Fall', 'Stand', 'Pain', 'Effort', 'Strong', 'Resilience', 'Motivational', 'Inspirational'],
    'Focus': ['Focus', 'Mind', 'Thoughts', 'Think', 'Best', 'Effort', 'Want', 'Goals', 'Process', 'Will', 'Success Is'],
    'Teamwork': ['Team', 'Together', 'Support', 'Friends', 'Game', 'Win']
}

#determine the focus based on the tags. consider cases where the input is a string and a list 
# return list of focuses, or 'general' if no focus is found
def det_focus(tags):
    if isinstance(tags, str):
        tag_lst = tags.lower().split(',')
    elif isinstance(tags, list):
        tag_lst = [tag.lower() for tag in tags]
    else: 
        return 'general'
    out = set()
    for tag in tag_lst:
        for focus, keywords in focus_mapping.items():
            if tag in(t.lower() for t in keywords):
                out.add(focus)
    return list(out) if out else ['general']

#add a focus column based on the tags 
filtered['focus'] = filtered['categories'].apply(det_focus).reset_index(drop=True)

#explode the dataframe such that each row contains a single focus
exploded = filtered.explode('focus', ignore_index=True)

#create 'training_text' column that will be used as input to our model
exploded['training_text'] = exploded.apply(lambda row: f"{str(row['focus']).strip()} Quote: {str(row['text']).strip()}", axis=1)

#set the training dataset to the trainig_text column
dataset = Dataset.from_pandas(exploded[['training_text']])

#initialize pretrained gpt2 model and tokenizer
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token                       #set padding token to end of sequence token
model = AutoModelForCausalLM.from_pretrained(model_name)        #for text generation 

#tokenize the training text
def tokenize_funct(batch):
    return tokenizer(batch['training_text'], truncation=True, padding=True, max_length=128, return_tensors ='pt')

tokenized = dataset.map(tokenize_funct, batched=True, remove_columns=dataset.column_names)

#create data collator to use at training
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#set training arguments
training_args = TrainingArguments(output_dir='./motivational_model', 
                                  overwrite_output_dir=True, 
                                  save_total_limit=2,
                                  per_device_train_batch_size=8, 
                                  per_device_eval_batch_size=4,  
                                  eval_strategy='steps', 
                                  eval_steps=200,
                                  num_train_epochs=5)

#set test size
train_test = tokenized.train_test_split(test_size=0.1)

#train the model useing model, arguments, data collator and tokenized data
trainer = Trainer(model = model, 
                  args = training_args, 
                  data_collator=data_collator, 
                  train_dataset= train_test['train'], 
                  eval_dataset=train_test['test'])

trainer.train()
trainer.save_model()
tokenizer.save_pretrained('./motivational_model')