
fineness = 50;
radius = 1.0;


function parabola(f,x) = ( 1/(4*f) ) * x*x; 

module egg(radius, exponent, steps){
    function my_pow(r,x) = r - r * x*x;
    function get_radius(x) = my_pow(radius,x);
    
    module go(x,z){
        translate([0,0,z])
            sphere(r=x, $fn=fineness);
        }
    
    step_size = 0.75 / steps ;
    my_range = [0:step_size:0.75];
    for (ii=my_range){
        go(get_radius(4*ii), ii);
    }
        
    
    }
    
radius1 = 0.125;
difference(){
    egg(radius1, 2.1, 40);
    translate([0,0,-0.1]) egg(radius1*0.99,2.1,40);
}
    
