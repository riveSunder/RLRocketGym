
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
    
    step_size = 3 / steps ;
    my_range = [0:step_size:3];
    for (ii=my_range){
        go(exponent*get_radius(ii), ii);
    }
        
    
    }
    
difference(){
    egg(1.0, 0.45, 40);
    translate([0,0,-0.3]) egg(1.0,0.45,40);
}
    