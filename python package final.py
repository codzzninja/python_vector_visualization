while True:
    print("\t\t\t VECTOR NOTATION\t\t\t\n")
    print("\t\t\t1.2D AND 3D VECTOR PLOTTING")
    print("\t\t\t2.ADDITION OF TWO VECTORS\n")
    print("\t\t\t3.SUBTRACTION OF TWO VECTORS\n")
    print("\t\t\t4.SCALAR OR DOT PRODUCT OF TWO VECTORS\n")
    print("\t\t\t5.VECTOR OR CROSS PRODUCT OF TWO VECTORS\n")
    print("\t\t\t6.SECTION FORMULA\n")
    print("\t\t\t7.PROJECTION OF VECTOR ON A LINE\n")
    print("\t\t\t8.MAGNITUDE OF VECTOR\n")
    print("\t\t\t9.NORMALIZATION OF VECTOR\n")
    print("\t\t\t10.SCALAR CONSTANT MULTIPLICATION\n")
    print("\t\t\t11.LINEAR COMBINATION\n")
    print("\t\t\t12.ANGLE BETWEEN TWO VECTORS\n")
    print("\t\t\t13.INTERSECTION OF VECTORS\n")
    print("\t\t\tENTER YOUR CHOICE\n")
    ch=int(input())


    if(ch==1):
        p=int(input("\nEnter 2 for 2d plotting or 3 for 3d plotting: "))
        if(p==3):
           
            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            input_str = input("\nEnter a comma-separated list of values: ")
            input_list = input_str.split(",")
            input_list = [float(x) for x in input_list]
            import numpy as np
            v = np.array(input_list)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.quiver(0, 0, 0, v[0], v[1], v[2], color='r', arrow_length_ratio=0.2)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim([0, 4])
            ax.set_ylim([0, 4])
            ax.set_zlim([0, 4])
            plt.show()
           
        if(p==2):
            import matplotlib.pyplot as plt
            input_str = input("\nEnter a comma-separated list of values: ")
            input_list = input_str.split(",")
            v = [float(x) for x in input_list]
            plt.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale = 1, color="Red")
            plt.xlim([-5, 5])
            plt.ylim([-5, 5])
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid()
            plt.show()
   
    elif(ch==2):
        t=int(input("\nEnter the 2d(2) or 3d(3) plotting: "))
        input_str = input("\nEnter a comma-separated list of values: ")
        input_list = input_str.split(",")
        input_list = [float(x) for x in input_list]
        import numpy as np
        a = np.array(input_list)
        input_str1 = input("\nEnter a comma-separated list of values: ")
        input_list1 = input_str1.split(",")
        input_list1 = [float(x) for x in input_list1]
        b = np.array(input_list1)
        res=np.add(a,b)
        print(res)
        if(t==2):
            import numpy as np
            import matplotlib.pyplot as plt
            def draw(x, y):
                xPlusy = (abs(x[0]+y[0]),abs(x[1]+y[1]))
                array = np.array([[0, 0, x[0], x[1]],
                              [x[0], x[1], y[0], y[1]],
                              [0, 0, xPlusy[0], xPlusy[1]]])
                print(array)
                X, Y, U, V = zip(*array)
                plt.figure()
                plt.ylabel('Y-axis')
                plt.xlabel('X-axis')
                ax = plt.gca()
                ax.quiver(X, Y, U, V, angles='xy', scale_units='xy',color=['r','b','g'],scale=1)
                ax.set_xlim([0, 20])
                ax.set_ylim([0, 20])
                plt.draw()
                plt.show()
            draw(a,b)
        elif(t==3):
            import matplotlib.pyplot as plt
            import numpy as np
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.quiver(0, 0, 0, a[0], a[1], a[2], color='r', arrow_length_ratio=0.1)
            ax.quiver(a[0], a[1], a[2], b[0], b[1], b[2], color='b', arrow_length_ratio=0.1)
            ax.quiver(0, 0, 0, res[0], res[1], res[2], color='b', arrow_length_ratio=0.1)
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.set_zlim([-10, 10])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.title('3D Vector Plot')
            plt.show()

    elif(ch==3):
        s=int(input("\nEnter the 2d(2) or 3d(3) plotting: "))
        input_str = input("\nEnter a comma-separated list of values: ")
        input_list = input_str.split(",")
        input_list = [float(x) for x in input_list]
        import numpy as np
        a = np.array(input_list)
        input_str1 = input("\nEnter a comma-separated list of values: ")
        input_list1 = input_str1.split(",")
        input_list1 = [float(x) for x in input_list1]
        b = np.array(input_list1)
        res=np.subtract(a,b)
        print(res)
        if(s==2):
            import numpy as np
            import matplotlib.pyplot as plt
            def draw(x, y):
                xPlusy = (x[0]-y[0],x[1]-y[1])
                array = np.array([[0, 0, x[0], x[1]],
                                 [x[0], x[1], -y[0], -y[1]],
                                 [0, 0, xPlusy[0], xPlusy[1]]])
                print(array)
                X, Y, U, V = zip(*array)
                print("X =",X)
                print("Y =",Y)
                print("U =",U)
                print("V =",V)
                plt.figure()
                plt.ylabel('Y-axis')
                plt.xlabel('X-axis')
                ax = plt.gca()
                ax.quiver(X, Y, U, V, angles='xy', scale_units='xy',color=['r','b','g'],scale=1)
                ax.set_xlim([-20, 20])
                ax.set_ylim([-20, 20])
                plt.draw()
                plt.show()
            draw(a,b)
        elif(s==3):
                import matplotlib.pyplot as plt
                import numpy as np
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.quiver(0, 0, 0, a[0], a[1], a[2], color='r', arrow_length_ratio=0.1)
                ax.quiver(a[0], a[1], a[2], -b[0], -b[1], -b[2], color='b', arrow_length_ratio=0.1)
                ax.quiver(0, 0, 0, res[0], res[1], res[2], color='b', arrow_length_ratio=0.1)
                ax.set_xlim([-10, 10])
                ax.set_ylim([-10, 10])
                ax.set_zlim([-10, 10])
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.title('3D Vector Plot')
                plt.show()
   
    elif(ch==4):
        z=int(input("\nEnter 1 for 1 dimensional and 2 for 2 dimensional: "))
        if(z==1):
            y1=int(input("\nEnter 2 for 2d or 3 for 3d: "))
            if(y1==2):
                import numpy as np
                a=eval(input("\nEnter the first vector: "))
                b=eval(input("\nEnter the second vector: "))
                dotproduct=np.dot(a,b)
                print('\nDot product is:', dotproduct)
            elif(y1==3):
                import numpy as n
                a=eval(input("\nEnter the first vector: "))
                b=eval(input("\nEnter the second vector: "))
                dotproduct = n.dot(a,b)
                print('\nDot product is:', dotproduct)
        elif(z==2):
            import numpy as np
            a=eval(input("\nEnter the first vector: "))
            b=eval(input("\nEnter the second vector: "))
            dotproduct = np.dot(a,b)
            print("\nDot product is: ",dotproduct)
           
    elif(ch==5):
        y=int(input("\nEnter the dimension of the array(1 or 2): "))
        if(y==1):
            y1=int(input("\nEnter the number of variables(2 or 3): "))
            if(y1==2):
                import numpy as np
                a=eval(input("\nEnter the first vector: "))
                b=eval(input("\nEnter the second vector: "))
                c=np.cross(a,b)
                print(c)
            elif(y1==3):
                import numpy as np
                a=eval(input("\nEnter the first vector: "))
                b=eval(input("\nEnter the second vector: "))
                c=np.cross(a,b)
                print(c)
        elif(y==2):
            y1=int(input("\nEnter the number of variables(2 or 3): "))
            if(y1==2):
                import numpy as np
                a=eval(input("\nEnter the first vector: "))
                b=eval(input("\nEnter the second vector: "))
                product = np.cross(a,b)
                print(product)
            elif(y1==3):
                import numpy as np
                a=eval(input("\nEnter the first vector: "))
                b=eval(input("\nEnter the second vector: "))
                product=np.cross(a,b)
                print(product)

    elif(ch==6):
        import numpy as np
        a_str = input("\nEnter the position vector of point A (comma separated): ")
        b_str = input("\nEnter the position vector of point B (comma separated): ")
        a = np.array([float(x.strip()) for x in a_str.split(",")])
        b = np.array([float(x.strip()) for x in b_str.split(",")])
        m = float(input("\nEnter the value of m: "))
        n = float(input("\nEnter the value of n: "))
        p = (m * b) + (n * a)
        print("\nThe coordinates of P are:", p)
   
    elif(ch==7):
        s=int(input("\nEnter 2 for 2d and 3 for 3d vector: "))
        import numpy as np
        import matplotlib.pyplot as plt
        input_str = input("\nEnter a comma-separated list of values: ")
        input_list = input_str.split(",")
        input_list = [float(x) for x in input_list]
        v = np.array(input_list)
        input_str1 = input("\nEnter a comma-separated list of values: ")
        input_list1 = input_str1.split(",")
        input_list1 = [float(x) for x in input_list1]
        input_vector1 = np.array(input_list1)
        magnitude=np.linalg.norm(input_vector1)
        proj=(np.dot(v,input_vector1)/(np.dot(input_vector1,input_vector1)))*magnitude
        proj1=(np.dot(v,input_vector1)/(np.dot(input_vector1,input_vector1)))*input_vector1
        print("\nProjection of 1st vector on second vector is: ",proj)
        if(s==2):
            fig, ax = plt.subplots()
            ax.set_xlim([0, 6])
            ax.set_ylim([0, 6])
            ax.arrow(0, 0, v[0], v[1], head_width=0.3, head_length=0.3, fc='k', ec='k')
            ax.arrow(0, 0, proj1[0], proj1[1], head_width=0.3, head_length=0.3, fc='r', ec='r')
            plt.show()
        elif(s==3):
            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.quiver(0, 0, 0, v[0], v[1], v[2], color='b', label='A')
            ax.quiver(0, 0, 0, input_vector1[0], input_vector1[1], input_vector1[2], color='r', label='B')
            ax.quiver(0, 0, 0, proj1[0], proj1[1], proj1[2], color='g', label='projB')
            ax.set_xlim([-1, 7])
            ax.set_ylim([-1, 7])
            ax.set_zlim([-1, 7])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            plt.show()
           
    elif(ch==8):
        import math
        x1=int(input("\nEnter the number of variables: "))
        if(x1==2):
            input_str = input("\nEnter a comma-separated list of values: ")
            input_list = input_str.split(",")
            input_list = [float(x) for x in input_list]
            import numpy as np
            input_vector = np.array(input_list)
            magnitude=np.linalg.norm(input_vector)
            print("\nMagnitude of the vector is: ",magnitude)
        elif(x1==3):
            input_str = input("\nEnter a comma-separated list of values: ")
            input_list = input_str.split(",")
            input_list = [float(x) for x in input_list]
            import numpy as np
            input_vector = np.array(input_list)
            magnitude=np.linalg.norm(input_vector)
            print("\nMagnitude of vector is: ",magnitude)
       
    elif(ch==9):
        input_str = input("\nEnter a comma-separated list of values: ")
        input_list = input_str.split(",")
        input_list = [float(x) for x in input_list]
        import numpy as np
        input_vector = np.array(input_list)
        magnitude=np.linalg.norm(input_vector)
        normalized=input_vector/magnitude
        print(normalized)
       
    elif(ch==10):
        input_str = input("\nEnter a 2d or 3d vector: ")
        input_list = input_str.split(",")
        input_list = [float(x) for x in input_list]
        import numpy as np
        input_vector = np.array(input_list)
        con=int(input("\nEnter the number to multiply: "))
        mul=con*(input_vector)
        print(mul)
       
    elif(ch==11):
        input_str = input("\nEnter a 2d or 3d vector: ")
        input_list1 = input_str.split(",")
        input_list1 = [float(x) for x in input_list1]
        import numpy as np
        input_vector1 = np.array(input_list1)
        input_str1 = input("\nIf the first vector is 2d enter 2d.If the first vector is 3d enter 3d ")
        input_list2 = input_str1.split(",")
        input_list2 = [float(x) for x in input_list2]
        import numpy as np
        input_vector2 = np.array(input_list2)
        con1=int(input("\nEnter the number to multiply for first vector: "))
        con2=int(input("\nEnter the number to multiply for second vector: "))
        res=con1*(input_vector1)+con2*(input_vector2)
        print(res)
       
       
    elif(ch==12):
        input_str = input("\nEnter a 2d or 3d vector")
        input_list1 = input_str.split(",")
        input_list1 = [float(x) for x in input_list1]
        import numpy as np
        input_vector1 = np.array(input_list1)
        input_str1 = input("\nIf the first vector is 2d enter 2d.if the first vector is 3d enter 3d: ")
        input_list2 = input_str1.split(",")
        input_list2 = [float(x) for x in input_list2]
        input_vector2 = np.array(input_list2)
        dot_product=np.dot(input_vector1,input_vector2)
        mag_v1=np.linalg.norm(input_vector1)
        mag_v2=np.linalg.norm(input_vector2)
        angle_rad=np.arccos(dot_product/(mag_v1*mag_v2))
        angle_deg=np.degrees(angle_rad)
        print(angle_deg)
       
    elif(ch==13):
        s=int(input("\nEnter 2 for 2d and 3 for 3d vector: "))
        input_str = input("\nEnter a 2d or 3d vector ")
        input_list1 = input_str.split(",")
        input_list1 = [float(x) for x in input_list1]
        import numpy as np
        a = np.array(input_list1)
        input_str1 = input("\nIf the first vector is 2d enter 2d.if the first vector is 3d enter 3d: ")
        input_list2 = input_str1.split(",")
        input_list2 = [float(x) for x in input_list2]
        b = np.array(input_list2)
        if(s==3):
            import numpy as np
            input_str=input("\nEnter the direction vector of first vector: ")
            input_list1 = input_str.split(",")
            input_list1 = [float(x) for x in input_list1]
            import numpy as np
            p = np.array(input_list1)
            input_str1 = input("\nEnter the dirction vector of second vector: ")
            input_list2 = input_str1.split(",")
            input_list2 = [float(x) for x in input_list2]
            q = np.array(input_list2)
            def vector_intersection(a, b, p, q):
                # Step 1: Find the cross product of the direction vectors
                cross = np.cross(p, q)    
                # Step 2: Check if the vectors are parallel
                if np.allclose(cross, 0):
                    return None
                # Step 3: Compute the position vector of the intersection point
                w = a - b
                t = np.dot(np.cross(w, q), np.cross(p, q)) / np.linalg.norm(np.cross(p, q))**2
                intersection = a + t * p
                return intersection
            intersection = vector_intersection(a, b, p, q)
            print(intersection)
        elif(s==2):
            intersection = np.linalg.solve([a, b], [0, 0])
            da = np.array([a[0], a[1]])
            db = np.array([b[0], b[1]])
            n = np.cross(da, db)
            if(n==0):
                print("\nVectors are parallel...")
            else:
                t = np.cross(b, -a)/ n
                p=a+t*n
            print("\nThe point of intersection of the two vectors is: ", p)

    print("\nPress yes to continue no to break:")
    w=input()
    if(w=="no"):
        break
    if(w=="yes"):
        continue