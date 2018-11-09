function y = max2loc(x)
   x_sorted = sort(x);
   x_sorted = abs(x_sorted);
   y = find(x == x_sorted(length(x)-1));
   if(length(y > 1))
       y = y(1);
   end
end