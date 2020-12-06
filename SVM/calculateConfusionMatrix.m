function confusionMatrix = calculateConfusionMatrix(actual, predicted)

      TP=0;
      FP=0;
      TN=0;
      FN=0;
       
      for rowIndex=1:height(actual)
          if(actual(rowIndex)==1 && predicted(rowIndex)==1)
              TP=TP+1;
          elseif(actual(rowIndex)==0 && predicted(rowIndex)==1)
              FP=FP+1;
          elseif(actual(rowIndex)==0 && predicted(rowIndex)==0)
              TN=TN+1;
          else
              FN=FN+1;
          end     
      end
      
      confusionMatrix.TP = TP;
      confusionMatrix.FP = FP;
      confusionMatrix.TN = TN;
      confusionMatrix.FN = FN;
end