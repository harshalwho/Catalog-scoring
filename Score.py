import pandas as pd
import streamlit as st
import datetime
import re

st.title('Catalogue Scoring App')

uploaded_file = st.file_uploader("Upload Catalogue CSV/Excel", type=['csv','xlsx']) 

if uploaded_file is not None:

  df = pd.read_csv(uploaded_file, encoding='latin-1') if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

  def clean_text(x):
     if isinstance(x, str): 
        return x.encode('utf-8', errors='replace').decode() 
     else:
        return x

  df = df.applymap(clean_text)

  with st.form('Enter the Catalogue details'):
    seller_name = st.text_input('Seller Name')
    catalogue_id = st.text_input('Catalogue ID')
    seller_email = st.text_input('Seller Email')
    contact_info = st.text_input('Contact Information')
    ai_desc = st.toggle('Perform in depth scoring for description content')
    ai_photo = st.toggle('Perform in depth scoring for product Images')
    genScore = st.form_submit_button('Generate Score')
  if genScore:
      if not seller_name or not seller_email or not catalogue_id:
          st.error('Please enter Seller Name, Catalogue ID and Seller Email')
      else:
        st.success('Your score is being calculated below')   
      # Initialize scores
        name_score = 0
        classification_score = 0 
        compliance_score = 0
        standard_score= 0
        desc_score= 0
        price_score= 0
        packaging_score= 0
        validity_score= 0
        indication_score= 0
        contraind_score= 0
        benefits_score= 0
        directions_score= 0
        composition_score= 0
        image_score= 0
        image_quality_score= 0
        video_score= 0
        stores_score = 0
        support_score= 0
        origin_score= 0
        practice_score= 0



        # Name score
        def get_name_score(x):
          if isinstance(x, str):
            if any(word in x.lower() for word in ['specific','particular']):
              return 2
            elif 'name' in x.lower():  
              return 1
            else:
              return 0
          else:
            return 0

        name_score = df['Product Name'].apply(get_name_score)

        # Classification score
        if 'Domain' in df.columns:
          classification_score += df['Domain'].notnull()
        if 'Sub Domain' in df.columns:
          classification_score += df['Sub Domain'].notnull()
        if 'Category' in df.columns:
          classification_score += df['Category'].notnull()
        if 'Sub Category' in df.columns:
          classification_score += df['Sub Category'].notnull()
        if 'Product Enum Code' in df.columns:  
          classification_score += df['Product Enum Code'].notnull()

        # Compliance score
        def get_compliance_score(x):         
          # Check if value is string
          if isinstance(x, str):
            # Search for FSSAI number
            fssai_pattern = r'\bFSSAI \d{14}\b'
            fssai_match = re.search(fssai_pattern, x)
            if fssai_match:
              # FSSAI number found, return 1
              return 1 
            
            # Search for CDSCO number  
            cdr_pattern = r'\bCDSCO-\w{4}-\d{3}\b'  
            cdr_match = re.search(cdr_pattern, x)
            
            if cdr_match:
              # CDSCO number found, return 1
              return 1

          # No compliance info found    
          return 0

        # Calculate score
        compliance_score = df['Manufacturer'].apply(get_compliance_score)

        # Standardization score
        if df.shape[1] > 0: 
          standard_score = df.apply(lambda x: len(set([y for y in map(str, x.values) if any(z in y.lower() for z in ['iso','ce','astm','halal'])])), axis=1)
      
        # Description score
        if 'Product Description' in df.columns:
          def get_desc_score(x):
            if isinstance(x, str):
              words = x.split()
              if len(words) >= 20 and len(words) <= 50:
                return 2
              elif len(words) > 0:  
                return 1
              else:
                return 0
            else:  
              return 0
            
          desc_score = df['Product Description'].apply(get_desc_score)

          # Product Description Scoring using BERT Embeddings
          if ai_desc:
            st.warning('Please be patient as AI protocol takes a while.')
            # Load pretrained BERT model
            from transformers import BertModel
            model = BertModel.from_pretrained('bert-base-uncased') 

            # Generate embeddings for product descriptions
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer('bert-base-uncased')
            desc_embeds = embedder.encode(df['Product Description'])

            # Get embeddings for ideal descriptions
            ideal_desc = ['High quality cotton t-shirt for men', 'Nutritious apple suitable for babies'] 
            ideal_embeds = embedder.encode(ideal_desc)

            # Calculate cosine similarity one pair at a time 
            from scipy.spatial.distance import cosine

            desc_scores = []
            for i in range(len(desc_embeds)):
              desc_vector = desc_embeds[i]
              sims = [cosine(desc_vector, ideal_vector) for ideal_vector in ideal_embeds]
              score = 1 - min(sims) # take max similarity
              desc_scores.append(score)

            # Map scores  
            desc_score = pd.Series(desc_scores).map(lambda x: 5 if x>=0.8 else 4 if x>=0.6 else 3 if x>=0.4 else 2 if x>=0.2 else 1)

        # Price score
        if 'MRP' in df.columns:
          price_score1 = df['MRP'].apply(lambda x: 1 if x>0 else -20 if x<0 else 0)
        if 'Max Price' in df.columns:
          price_score1 = df['Max Price'].apply(lambda x: 1 if x>0 else -20 if x<0 else 0)
        if 'Maximum Retail Price' in df.columns:
          price_score1 = df['Maximum Retail Price'].apply(lambda x: 1 if x>0 else -20 if x<0 else 0)
        if 'Price' in df.columns:
          price_score2 = df['Price'].apply(lambda x: 1 if x>0 else -20 if x<0 else 0)
        if 'Selling Price' in df.columns:
          price_score2 = df['Selling Price'].apply(lambda x: 1 if x>0 else -20 if x<0 else 0)
        if 'Quoted Price' in df.columns:
          price_score2 = df['Quoted Price'].apply(lambda x: 1 if x>0 else -20 if x<0 else 0)
        price_score = price_score + price_score2

        # Packaging score
        packaging_cols = ['Pack Quantity','Pack Size','UOM','Net Quantity']
        packaging_score = 0 
        for col in packaging_cols:
          if col in df.columns:
            packaging_score += df[col].notnull()

        # Validity score
        validity_score = 0
        if 'Expiry' in df.columns or 'Best Before' in df.columns:
            validity_score = 1

        # Indication score
        if 'Product Description' in df.columns:
          def get_indication_score(x):
            if isinstance(x, str):
              if any(word in x.lower() for word in ['kids','children','babies','age']):
                return 1
              else: 
                return 0
            
            else:
              return 0
          indication_score = df['Product Description'].apply(get_indication_score)          

        # Contraindication score
        if 'Product Description' in df.columns:
          def get_contraind_score(x):
            if isinstance(x, str):
              if any(word in x.lower() for word in ['avoid','not for']):
                return 1
              else:
                return 0
            
            else:
              return 0

          contraind_score = df['Product Description'].apply(get_contraind_score)
        # Benefits score
        if 'Product Description' in df.columns:          
          def get_benef_score(x):
            if isinstance(x, str):
              if any(word in x.lower() for word in ['useful for','enhances', 'helpful', 'beneficial']):
                return 1
              else:
                return 0            
            else:
              return 0
          benefits_score = df['Product Description'].apply(get_benef_score)

        # Directions score
        # if 'Product Description' in df.columns:  
        #   directions_score = df['Product Description'].apply(lambda x: 2 if len(re.findall(r'\d+[\.\)]',x))>=3 else 1 if len(re.findall(r'\d+[\.\)]',x))>0 else 0)
        if 'Product Description' in df.columns:          
          def get_dir_score(x):
            if isinstance(x, str):
              if any(word in x.lower() for word in ['step','mix', 'measure', 'apply']):
                return 1
              else:
                return 0            
            else:
              return 0
          directions_score1 = df['Product Description'].apply(get_dir_score)
          def get_directions_score(x):

            if isinstance(x, str):           
              matches = re.findall(r'\d+[\.\)]', x)              
              if len(matches) >= 3:
                return 2
              elif len(matches) > 0:
                return 1
              else:
                return 0
            else:
              return 0            
          directions_score2 = df['Product Description'].apply(get_directions_score)
          directions_score = directions_score1 + directions_score2
        # Composition score
        if 'Ingredients' in df.columns:
          
          def get_composition_score(x):

            if isinstance(x, str):
              ingredients = x.split(',')
              return len(ingredients)

            else:
              return 0

          composition_score = df['Ingredients'].apply(get_composition_score)

        # Image score
        image_cols = [col for col in df.columns if 'Image' in col]
        image_score = len(image_cols)

        image_quality_score = 2 #placeholder
        if ai_photo:
          st.warning('Please wait more as Image processing protocol takes significant time.')
          # ResNet image score 

          import torch
          import torchvision.models as models
          import torchvision.transforms as transforms
          import numpy as np
          import requests
          from PIL import Image
          from io import BytesIO

          # Load model
          resnet = models.resnet18(pretrained=True)
          resnet.eval()

          # Transform 
          transform = transforms.Compose([
              transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ])

          # Get image columns
          image_cols = [col for col in df.columns if 'image' in col]

          if len(image_cols) > 0:

            # Extract features
            image_features = []

            for col in image_cols:

              urls = df[col]

              for url in urls:

                try:
                  response = requests.get(url)
                  img = Image.open(BytesIO(response.content))

                  img_transformed = transform(img)
                  batch_t = torch.unsqueeze(img_transformed, 0)
                  feat = resnet(batch_t)
                  
                  # Make 2D
                  feat = np.squeeze(feat.detach().numpy())
                  feat = np.expand_dims(feat, axis=0)
                  
                  image_features.append(feat)

                except Exception as e:
                  print(f"Error downloading {url}: {e}")
                  
            if len(image_features) > 0:
              X = np.concatenate(image_features, axis=0) 
              y = np.array([1 if 'good' in col else 0 for col in image_cols])
              y = np.expand_dims(y,1)


            else:
              print("No valid images found")
              
          else:
            print("No image columns found")

        # Video score
        if 'Video' in df.columns:
          video_score = 1

        # Stores score
        stores_cols = ['Warehouse','Service Area']
        stores_score = 0
        for col in stores_cols:
          if col in df.columns:
            stores_score += df[col].notnull()

        # Support score
        if 'Customer Care Contact' in df.columns:
          support_score = 1

        # Origin score
        if 'Country Of Origin' in df.columns:
          origin_score = 1  

        # Practice score
        practice_cols = ['Time to Ship','Returnable','Cancellable','COD','Payment Terms']
        for col in practice_cols:
          if col in df.columns:
            practice_score += df[col].notnull()

        # Calculate total score
        if df.shape[0] > 0:
          df['Total Score'] = name_score + classification_score + compliance_score + standard_score + desc_score + price_score + packaging_score + validity_score + indication_score + contraind_score + benefits_score + directions_score + composition_score + image_score + image_quality_score + video_score + stores_score + support_score + origin_score + practice_score
          # Add image score 
          if 'Image Score' in df.columns:
            df['Total Score'] += df['Image Score']

          catalogue_score = df['Total Score'].mean()

        # Save results    
        today = datetime.date.today().strftime('%Y-%m-%d')
        results_df = pd.DataFrame({'Seller Name': [seller_name], 
                                'Catalogue ID': [catalogue_id],
                                'Seller Email': [seller_email],
                                'Date': [today],
                                'Score': [catalogue_score]})
                                
        results_df.to_csv('catalogue_scores.csv', mode='a', header=False, index=False)

        st.write(f'The catalogue score is: {catalogue_score}')
